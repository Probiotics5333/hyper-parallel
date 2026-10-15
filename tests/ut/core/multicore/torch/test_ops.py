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

"""Unit tests for native Torch adapter registration diagnostics."""

import ast
import inspect
import json
from pathlib import Path
import re
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.core.multicore._loader import NativeComponentUnavailableError
from hyper_parallel.core.multicore.torch import ops


class TestTorchOps(unittest.TestCase):
    """Native initialization is cached, but Torch itself is imported normally."""

    def tearDown(self) -> None:
        """Never retain a mocked native-registration cache between tests."""
        ops._load_native.cache_clear()

    def test_adapter_failure_reports_original_cause(self):
        """Preserve the failed library and ABI error in the native diagnostic."""
        with (
            patch.object(ops, "get_multicore_paths", return_value=(Path("vendor"), Path("bad.so"))),
            patch.object(ops, "preload_vendor_library"),
            patch.object(ops.torch.ops, "load_library", side_effect=OSError("bad ABI")),
            self.assertRaisesRegex(NativeComponentUnavailableError, "ADAPTER-LOAD-FAILED.*bad ABI"),
        ):
            ops._load_native()

    def test_successful_registration_is_cached(self):
        """Load the adapter only once across repeated operation calls."""
        with (
            patch.object(ops, "get_multicore_paths", return_value=(Path("vendor"), Path("good.so"))),
            patch.object(ops, "preload_vendor_library") as preload,
            patch.object(ops.torch.ops, "load_library") as load,
        ):
            ops._load_native()
            ops._load_native()
        preload.assert_called_once_with(Path("vendor"))
        load.assert_called_once_with("good.so")

    def test_diagnostic_buffer_keyword_reaches_forward_and_backward_native_ops(self):
        """Forward either diagnostic tool's buffer through the unchanged ABI slot."""
        for op_name, buffer_index in (("mega_moe", 22), ("mega_moe_grad", 29)):
            with self.subTest(op_name=op_name):
                wrapper = getattr(ops, f"{op_name}_with_diagnostic_buffer")
                arguments = {name: object() for name in inspect.signature(wrapper).parameters}
                arguments["diagnostic_buffer"] = torch.zeros(64, dtype=torch.uint8)
                with (
                    patch.object(ops, "_load_native") as load,
                    patch.object(ops.torch.ops.hyper_parallel, op_name, create=True) as native,
                ):
                    wrapper(**arguments)
                load.assert_called_once_with()
                native.assert_called_once_with(*arguments.values())
                self.assertIs(native.call_args.args[buffer_index], arguments["diagnostic_buffer"])

    def test_public_wrappers_keep_the_unused_diagnostic_placeholder(self):
        """Do not add a diagnostic argument to the ordinary public API."""
        for op_name, buffer_index in (("mega_moe", 22), ("mega_moe_grad", 29)):
            with self.subTest(op_name=op_name):
                wrapper = getattr(ops, op_name)
                arguments = {name: object() for name in inspect.signature(wrapper).parameters}
                self.assertNotIn("diagnostic_buffer", arguments)
                expected = list(arguments.values())
                expected.insert(buffer_index, arguments["all_event_counters"])
                with (
                    patch.object(ops, "_load_native"),
                    patch.object(ops.torch.ops.hyper_parallel, op_name, create=True) as native,
                ):
                    wrapper(**arguments)
                native.assert_called_once_with(*expected)
                self.assertIs(native.call_args.args[buffer_index], arguments["all_event_counters"])

    def test_torch_schema_matches_diagnostic_wrapper_names_and_order(self):
        """Parse actual C++ schemas on CPU to detect keyword or position drift."""
        source = Path(ops.__file__).parent / "csrc" / "registration.cpp"
        definitions = re.findall(r"m\.def\((.*?)\);", source.read_text(encoding="utf-8"), flags=re.DOTALL)
        self.assertEqual(len(definitions), 2)
        for definition in definitions:
            literals = re.findall(r'"(?:[^"\\]|\\.)*"', definition)
            schema = torch._C.parse_schema("".join(ast.literal_eval(literal) for literal in literals))
            with self.subTest(op_name=schema.name):
                wrapper = getattr(ops, f"{schema.name}_with_diagnostic_buffer")
                self.assertEqual(
                    [argument.name for argument in schema.arguments],
                    list(inspect.signature(wrapper).parameters),
                )

    def test_cann_input_metadata_and_worker_indices_match_the_diagnostic_abi(self):
        """Keep both CANN targets and worker lists on the shared buffer contract."""
        multicore_root = Path(ops.__file__).parents[1]
        for op_name, abi_index, worker_index in (("hyper_mega_moe", 22, 25), ("hyper_mega_moe_grad", 29, 32)):
            with self.subTest(op_name=op_name):
                op_root = multicore_root / "ops" / op_name
                proto = (op_root / "op_graph" / f"{op_name}_proto.h").read_text(encoding="utf-8")
                definition = (op_root / "op_host" / f"{op_name}_def.cpp").read_text(encoding="utf-8")
                self.assertEqual(re.findall(r"\.INPUT\((\w+),", proto)[abi_index], "diagnostic_buffer")
                self.assertEqual(re.findall(r'this->Input\("(\w+)"\)', definition)[abi_index], "diagnostic_buffer")
                worker = (op_root / "op_kernel" / "worker_kernel.cpp").read_text(encoding="utf-8")
                self.assertRegex(worker, rf"DIAGNOSTIC_IDX\s*=\s*{worker_index};")
                kernel = (op_root / "op_kernel" / f"{op_name}.cpp").read_text(encoding="utf-8")
                input_list = re.search(r"GM_ADDR input_list\[\]\s*=\s*\{(.*?)\}", kernel, flags=re.DOTALL)
                self.assertIsNotNone(input_list)
                self.assertEqual(input_list.group(1).split(",")[worker_index].strip(), "diagnostic_buffer")
                for target in ("ascend910b", "ascend910_93"):
                    config_path = op_root / "op_host" / "config" / target / f"{op_name}_binary.json"
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                    for variant in config["op_list"]:
                        buffer_input = variant["inputs"][abi_index]
                        self.assertEqual(buffer_input["name"], "diagnostic_buffer")
                        self.assertEqual(buffer_input["index"], abi_index)
                        self.assertEqual(buffer_input["dtype"], "uint8")
