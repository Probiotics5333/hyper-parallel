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
"""Shared graph identity and independent profiler runtime tests."""

import struct
import unittest

from hyper_parallel.core.multicore.diagnostics.common.graph import DiagnosticSpec, build_graph_metadata
from hyper_parallel.core.multicore.diagnostics.common.runtime import PreparedDiagnosticRuntime
from hyper_parallel.core.multicore.diagnostics.profiler.cycle_trace import (
    GRAPH_STAGE_DESC_BASE, MAX_PROFILE_BUFFER_BYTES, _PreparedProfileRuntime,
    _calculate_profile_layout, _configure_trace_metadata, _resolve_cycle_frequency_mhz,
)
from hyper_parallel.core.multicore.scheduler.config import RuntimeConfigC, TaskDescC, TaskType
from hyper_parallel.core.multicore.scheduler.graph import ComputeGraph, OperatorNode, OpType, SplitSpec
from hyper_parallel.core.multicore.scheduler.runtime import allocate_runtime_config


def _runtime_config(task_capacity: int = 16) -> RuntimeConfigC:
    """Allocate an actual graph-sized Host configuration."""
    return allocate_runtime_config(task_capacity)


def _three_record_task() -> TaskDescC:
    """Create a task with a dependency and completion event."""
    task = TaskDescC()
    task.task_type = TaskType.TASK_GROUPED_MATMUL
    task.dependent_event = 0
    return task


def _graph():
    """Create generic stage identity without any expert semantics."""
    graph = ComputeGraph()
    for name, count in (("first", 1), ("second", 2)):
        operator = OperatorNode(name, OpType.SWIGLU, [], [], [], 1,
                                SplitSpec(None, lambda _: 1), -1, None)
        operator.task_num = count
        graph.add_op(operator)
    return graph


class TestDiagnosticMetadata(unittest.TestCase):
    """Distinguish Host identity from profiling-specific serialized IDs."""

    def test_graph_metadata_is_read_only(self):
        """Resolving task identity must never annotate the base TaskDesc."""
        config = _runtime_config()
        original = bytes(config)
        metadata = build_graph_metadata(_graph(), config, DiagnosticSpec("Custom", "Shard", lambda *_: 7))
        self.assertEqual(metadata.stage_names, {0: "First", 1: "Second"})
        self.assertEqual(metadata.task_stage_ids, {0: 0, 1: 1, 2: 1})
        self.assertEqual(metadata.task_owner_ids, {0: 7, 1: 7, 2: 7})
        self.assertEqual(bytes(config), original)
        with self.assertRaises(TypeError):
            metadata.stage_names[0] = "changed"

    def test_repeated_names_have_distinct_stage_ids(self):
        """Profiling supports repeated display names without conflating task identity."""
        graph = _graph()
        for operator in graph.topological_sort():
            operator.diagnostic_name = "Repeated"
        metadata = build_graph_metadata(graph, _runtime_config(), DiagnosticSpec("Custom"))
        self.assertEqual(metadata.stage_names, {0: "Repeated", 1: "Repeated"})

    def test_optional_and_invalid_owners(self):
        """Allow missing owners but reject IDs outside the serializable uint32 range."""
        graph, config = _graph(), _runtime_config()
        self.assertIsNone(build_graph_metadata(graph, config, DiagnosticSpec("Custom")).task_owner_ids[0])
        for owner in (-1, True, 0xFFFFFFFF, "owner"):
            with self.subTest(owner=owner), self.assertRaisesRegex(ValueError, "owner_id"):
                build_graph_metadata(graph, config, DiagnosticSpec("Custom", owner_resolver=lambda *_: owner))

    def test_trace_encoding_uses_a_private_configuration(self):
        """Only profiling owns profile_desc_id and profile_owner_id."""
        config = _runtime_config()
        config.task_num = 3
        metadata = build_graph_metadata(_graph(), config, DiagnosticSpec("Custom", "Shard", lambda *_: 7))
        original = bytes(config)
        variant = type(config).from_buffer_copy(original)
        trace = _configure_trace_metadata(variant, metadata)
        self.assertEqual(variant.all_tasks[0].profile_desc_id, GRAPH_STAGE_DESC_BASE)
        self.assertEqual(variant.all_tasks[2].profile_desc_id, GRAPH_STAGE_DESC_BASE + 1)
        self.assertEqual(variant.all_tasks[2].profile_owner_id, 7)
        self.assertEqual(trace.task_stage_names[2], "Second")
        self.assertEqual(bytes(config), original)

    def test_sparse_termination_preserves_unrelated_slots(self):
        """Queue-only system tasks do not cause annotation of unused task slots."""
        config = _runtime_config()
        config.task_index_num[0] = 1
        config.cube_task_indices[0] = 7
        config.all_tasks[7].task_type = TaskType.TASK_TERMINATE
        config.all_tasks[1].profile_desc_id = 0xABCDEF
        metadata = build_graph_metadata(ComputeGraph(), config, DiagnosticSpec("Sparse"))
        _configure_trace_metadata(config, metadata)
        self.assertEqual(config.all_tasks[7].profile_desc_id, 0x10006)
        self.assertEqual(config.all_tasks[1].profile_desc_id, 0xABCDEF)

    def test_only_selected_profiler_prepares_a_trace_image(self):
        """Normal preparation produces exactly one normal image and no trace state."""
        config = _runtime_config()
        config.num_workers = 2
        created = []
        def tensor_factory(data: bytes) -> bytes:
            """Count serialized images without creating Device resources."""
            created.append(data)
            return data
        runtime = PreparedDiagnosticRuntime(ComputeGraph(), config, DiagnosticSpec("Custom"),
                                            parameter_count=0, tensor_factory=tensor_factory, rank=3, device_id=7)
        self.assertEqual(len(created), 1)
        self.assertIsNone(runtime._metadata)
        original = bytes(config)
        profile = _PreparedProfileRuntime(runtime)
        self.assertEqual(len(created), 2)
        offset = RuntimeConfigC.diagnostic_mode.offset
        self.assertEqual(struct.unpack_from("<I", runtime.normal_tensor, offset)[0], 0)
        self.assertEqual(struct.unpack_from("<I", profile.profile_tensor, offset)[0], 1)
        self.assertEqual(bytes(config), original)
        self.assertEqual((profile.rank, profile.device_id), (3, 7))


class TestMegaKernelProfileLayout(unittest.TestCase):
    """Validate exact bounded AIC/AIV buffer sizing."""

    def test_layout_rounds_busiest_workers_to_sixteen_records(self):
        """Mirror Device round-robin assignment for AIC and AIV independently."""
        runtime_config = _runtime_config(128)
        runtime_config.num_workers = 48
        runtime_config.all_tasks[0] = _three_record_task()
        runtime_config.task_index_num[0] = 121
        runtime_config.task_index_num[1] = 24

        layout = _calculate_profile_layout(runtime_config)

        self.assertEqual(layout.aic_required_records, 18)
        self.assertEqual(layout.aiv_required_records, 3)
        self.assertEqual(layout.aic_record_capacity, 32)
        self.assertEqual(layout.aiv_record_capacity, 16)
        self.assertEqual(layout.buffer_size, 53760)
        self.assertLess(layout.buffer_size, MAX_PROFILE_BUFFER_BYTES)

    def test_layout_caps_each_worker_at_256_records(self):
        """Bound Device memory when a long schedule needs more records."""
        runtime_config = _runtime_config(2048)
        runtime_config.num_workers = 48
        runtime_config.all_tasks[0] = _three_record_task()
        runtime_config.task_index_num[0] = 2041

        layout = _calculate_profile_layout(runtime_config)

        self.assertEqual(layout.aic_required_records, 258)
        self.assertEqual(layout.aic_record_capacity, 256)

    def test_soc_aliases_use_the_documented_counter_frequency(self):
        """Recognize current 910B and 910C/A3 Torch NPU names."""
        for soc_name in ("Ascend910B4", "Ascend910C1", "Ascend910_9372"):
            with self.subTest(soc_name=soc_name):
                self.assertEqual(_resolve_cycle_frequency_mhz(soc_name), 50.0)


if __name__ == "__main__":
    unittest.main()
