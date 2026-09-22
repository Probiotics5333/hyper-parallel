# Copyright 2026 Huawei Technologies Co., Ltd.
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
"""Unit tests for the HyperMegaMhcGrad scheduling graph."""

import unittest
from unittest.mock import patch

from hyper_parallel.core.multicore.modules.mega_mhc_grad.function import _resolve_core_counts
from hyper_parallel.core.multicore.modules.mega_mhc_grad.graph import (
    build_mega_mhc_grad_graph,
    order_vector_tasks_by_stage,
)
from hyper_parallel.core.multicore.scheduler.builder import build_runtime_config
from hyper_parallel.core.multicore.scheduler.config import TaskType


class TestMegaMhcGradGraph(unittest.TestCase):
    """Validate scheduling invariants that protect the native reduction buffers."""

    @patch("hyper_parallel.core.multicore.modules.mega_mhc_grad.function.torch.npu.get_device_limit")
    def test_resolve_core_counts_uses_device_vector_count(self, mock_get_device_limit):
        """Device AIC and AIV counts must remain distinct runtime inputs."""
        mock_get_device_limit.return_value = {
            "cube_core_num": 20,
            "vector_core_num": 40,
        }

        self.assertEqual(_resolve_core_counts(0), (20, 40))

    @patch("hyper_parallel.core.multicore.modules.mega_mhc_grad.function.torch.npu.get_device_limit")
    def test_resolve_core_counts_rejects_incompatible_mixed_ratio(self, mock_get_device_limit):
        """The native 1:2 mixed kernel must reject an incompatible resource limit."""
        mock_get_device_limit.return_value = {
            "cube_core_num": 20,
            "vector_core_num": 32,
        }

        with self.assertRaisesRegex(RuntimeError, "KERNEL_TYPE_MIX_AIC_1_2"):
            _resolve_core_counts(0)

    def test_reduction_output_initialization_uses_all_vector_workers(self):
        """The 4096x5120 plan must shard output clearing over every AIV."""
        num_cube_cores = 20
        num_vector_cores = 40
        graph, topology = build_mega_mhc_grad_graph(
            4096,
            5120,
            num_vector_cores=num_vector_cores,
        )
        config = build_runtime_config(
            graph,
            topology,
            num_cube_cores=num_cube_cores,
        )
        order_vector_tasks_by_stage(config)

        self.assertEqual(int(config.num_workers), num_vector_cores)

        vector_task_count = int(config.task_index_num[1])
        vector_tasks = [
            config.all_tasks[int(config.vector_task_indices[index])]
            for index in range(vector_task_count)
        ]
        init_tasks = [
            task
            for task in vector_tasks
            if TaskType(task.task_type) == TaskType.TASK_MHC_GRAD_PREV_A
        ]

        self.assertEqual(len(init_tasks), num_vector_cores)
        self.assertEqual(
            [int(task.task_index) for task in init_tasks],
            list(range(num_vector_cores)),
        )
        self.assertTrue(
            all(int(task.task_split_num) == num_vector_cores for task in init_tasks)
        )
        self.assertEqual(int(config.all_event_num_triggers[0]), num_vector_cores)


if __name__ == "__main__":
    unittest.main()
