# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Unit tests for schedule-driven Torch MegaKernel Tensor dump."""

import json
from pathlib import Path
import tempfile
import unittest

import torch

from hyper_parallel.core.multicore.diagnostics import prepare_mega_kernel_diagnostics, schedule
from hyper_parallel.core.multicore.diagnostics import profiler as torch_profiler
from hyper_parallel.core.multicore.scheduler.runtime import allocate_runtime_config
from hyper_parallel.core.multicore.diagnostics.dump import api as tensor_dump
from hyper_parallel.core.multicore.diagnostics.common.session import _reset_active_session_for_test
from hyper_parallel.core.multicore.scheduler.graph import (
    ComputeGraph,
    OperatorNode,
    OpType,
    SplitSpec,
    TensorSpec,
)


def _make_test_graph() -> ComputeGraph:
    """Build a two-stage graph with all three L0 Tensor roles."""
    graph_input = TensorSpec("input", [2], param_position=0)
    intermediate = TensorSpec("intermediate", [2], param_position=1)
    graph_output = TensorSpec("output", [2], param_position=2)
    first = OperatorNode(
        "up_proj", OpType.GMM, [graph_input], [intermediate], [0, 1], 1,
        SplitSpec(None, lambda _: 1, [0]), -1, None, diagnostic_name="GMM1",
    )
    second = OperatorNode(
        "down_proj", OpType.GMM, [intermediate], [graph_output], [1, 2], 1,
        SplitSpec([(0, 0)], lambda _: 1, [0]), -1, None, diagnostic_name="GMM2",
    )
    return ComputeGraph().add_op(first).add_op(second).add_edge(first, second)


class TestTensorDumpConfig(unittest.TestCase):
    """Validate L0 configuration normalization and filtering."""

    def test_normalizes_scalar_filters(self):
        config = tensor_dump.TensorDumpConfig(
            "dump", kernel_names="mega_moe", tensor_names="output", roles="final_output",
            directions="forward", ranks=2,
        )

        self.assertEqual(config.tensor_names, ("output",))
        self.assertEqual(config.kernel_names, ("mega_moe",))
        self.assertEqual(config.roles, ("final_output",))
        self.assertEqual(config.directions, ("forward",))
        self.assertEqual(config.ranks, (2,))
        self.assertTrue(config.selects_invocation("forward", 2, "mega_moe"))
        self.assertFalse(config.selects_invocation("forward", 2, "other_kernel"))
        self.assertFalse(config.selects_invocation("backward", 2))

    def test_normalizes_l1_filters(self):
        config = tensor_dump.TensorDumpConfig(
            "dump",
            level="l1",
            stages="SwiGLU",
            task_ids=3,
            stage_task_indices=1,
            owner_ids=2,
            core_types="AIV",
            core_ids=0,
            io="output",
        )

        self.assertEqual(config.level, "L1")
        self.assertEqual(config.stages, ("SwiGLU",))
        self.assertEqual(config.task_ids, (3,))
        self.assertEqual(config.core_types, ("AIV",))
        self.assertEqual(config.io, ("output",))

    def test_rejects_unsupported_level_and_invalid_rank(self):
        with self.assertRaisesRegex(ValueError, "unsupported Tensor dump level"):
            tensor_dump.TensorDumpConfig("dump", level="L2")
        with self.assertRaisesRegex(ValueError, "requires at least one"):
            tensor_dump.TensorDumpConfig("dump", level="L1")
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            tensor_dump.TensorDumpConfig("dump", ranks=True)


class TestGraphDiagnosticMetadata(unittest.TestCase):
    """Validate generic and explicitly overridden operator stage names."""

    def test_default_stage_names_do_not_depend_on_operator_type(self):
        operators = _make_test_graph().topological_sort()
        for operator in operators:
            operator.diagnostic_name = None

        self.assertEqual(
            tensor_dump._stage_names(operators),
            {"up_proj": "UpProj", "down_proj": "DownProj"},
        )

    def test_rejects_duplicate_explicit_stage_names(self):
        operators = _make_test_graph().topological_sort()
        operators[1].diagnostic_name = "GMM1"

        with self.assertRaisesRegex(ValueError, "stage directory"):
            tensor_dump._stage_names(operators)


class TestTorchMegaKernelTensorDumper(unittest.TestCase):
    """Validate capture boundaries, metadata, filters and diagnostic isolation."""

    def tearDown(self) -> None:
        """Reset the process-local diagnostic session after each test."""
        _reset_active_session_for_test()

    def test_exports_graph_tensors(self):
        graph = _make_test_graph()
        parameters = [torch.tensor([1.0, 2.0]), torch.zeros(2), torch.zeros(2)]
        with tempfile.TemporaryDirectory() as output_dir:
            config = tensor_dump.TensorDumpConfig(
                output_dir, kernel_names="test_kernel", directions="forward", ranks=0,
            )
            capture_schedule = schedule(wait=0, warmup=0, active=1, repeat=1)
            with tensor_dump.tensor_dump(config, schedule=capture_schedule) as dumper:
                adapter = prepare_mega_kernel_diagnostics(
                    graph, allocate_runtime_config(0), kernel_name="test_kernel", parameter_count=3, device="cpu",
                )
                call = adapter.launch(parameters, direction="forward", fallback_event_counters=torch.zeros(16))
                parameters[1].copy_(parameters[0] + 1)
                parameters[2].copy_(parameters[1] * 2)
                call.complete()
                dumper.step()

            manifest_path = dumper.last_manifest_paths[0]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            invocation = manifest["invocations"][0]
            tensors = {item["name"]: item for item in invocation["tensors"]}
            self.assertEqual(manifest["schemaVersion"], 4)
            self.assertEqual(manifest["tensorSerialization"], "torch.save")
            self.assertEqual(manifest["tensorFileLayout"], "stage_only")
            self.assertEqual(manifest["capturePhaseSemantics"], "mega_kernel_boundary")
            self.assertEqual(manifest["stageViewStorage"], "primary_file_with_hard_links")
            self.assertEqual(invocation["direction"], "forward")
            self.assertEqual(invocation["kernelName"], "test_kernel")
            self.assertEqual(tensors["input"]["capturePhase"], "before")
            self.assertEqual(tensors["intermediate"]["role"], "stage_output")
            self.assertEqual(tensors["output"]["role"], "final_output")
            self.assertEqual(
                [(port["stage"], port["io"], port["index"]) for port in tensors["intermediate"]["ports"]],
                [("GMM1", "output", 0), ("GMM2", "input", 0)],
            )
            invocation_dir = manifest_path.parent / "invocation_000000_forward"
            self.assertFalse((invocation_dir / "before").exists())
            self.assertFalse((invocation_dir / "after").exists())
            for metadata in tensors.values():
                primary_path = manifest_path.parent / metadata["file"]
                self.assertEqual(metadata["file"], metadata["ports"][0]["file"])
                for port in metadata["ports"]:
                    stage_path = manifest_path.parent / port["file"]
                    self.assertTrue(stage_path.is_file())
                    self.assertTrue(primary_path.samefile(stage_path))
            torch.testing.assert_close(
                torch.load(
                    manifest_path.parent / tensors["output"]["file"],
                    map_location="cpu",
                    weights_only=True,
                ),
                torch.tensor([4.0, 6.0]),
            )

    def test_filters_and_preserves_bfloat16(self):
        graph = _make_test_graph()
        parameters = [torch.ones(2, dtype=torch.bfloat16) for _ in range(3)]
        with tempfile.TemporaryDirectory() as output_dir:
            config = tensor_dump.TensorDumpConfig(
                output_dir, tensor_names="input", roles="external_input", directions="forward",
            )
            with tensor_dump.tensor_dump(config) as dumper:
                adapter = prepare_mega_kernel_diagnostics(
                    graph, allocate_runtime_config(0), kernel_name="test_kernel", parameter_count=3, device="cpu",
                )
                call = adapter.launch(parameters, direction="forward", fallback_event_counters=torch.zeros(16))
                call.complete()

            manifest_path = dumper.last_manifest_paths[0]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            metadata = manifest["invocations"][0]["tensors"][0]
            payload = torch.load(
                manifest_path.parent / metadata["file"],
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(metadata["originalDtype"], "bfloat16")
            self.assertEqual(metadata["storedDtype"], "bfloat16")
            self.assertEqual(payload.dtype, torch.bfloat16)
            torch.testing.assert_close(payload, torch.ones(2, dtype=torch.bfloat16))
            self.assertEqual(manifest["invocations"][0]["tensorCount"], 1)

    def test_exports_unmapped_tensor_under_kernel_boundary_stage(self):
        """Keep an unmapped diagnostic Tensor in the stage-only public layout."""
        graph_tensor = tensor_dump._GraphTensor(
            name="auxiliary",
            param_position=3,
            declared_shape=(2,),
            role="external_input",
            producers=(),
            consumers=(),
            ports=(),
        )
        snapshot = tensor_dump._snapshot_tensor(torch.tensor([1.0, 2.0]), graph_tensor, "before")
        with tempfile.TemporaryDirectory() as output_dir:
            window_dir = Path(output_dir)
            invocation_dir = window_dir / "invocation_000000_forward"

            output_path, ports = tensor_dump._write_l0_stage_files(
                snapshot,
                invocation_dir,
                window_dir,
                "auxiliary",
            )

            expected_path = invocation_dir / "stages" / "__kernel_boundary__" / "input_03_auxiliary.pt"
            self.assertEqual(output_path, expected_path)
            self.assertEqual(ports, [])
            torch.testing.assert_close(
                torch.load(output_path, map_location="cpu", weights_only=True),
                torch.tensor([1.0, 2.0]),
            )

    def test_profiler_and_tensor_dump_are_mutually_exclusive(self):
        config = tensor_dump.TensorDumpConfig("dump")

        with torch_profiler.mega_kernel_profile():
            with self.assertRaisesRegex(RuntimeError, "nested"):
                with tensor_dump.tensor_dump(config):
                    pass
