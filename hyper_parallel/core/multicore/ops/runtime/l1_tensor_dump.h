/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MULTICORE_SCHEDULER_L1_TENSOR_DUMP_H
#define MULTICORE_SCHEDULER_L1_TENSOR_DUMP_H

#include "kernel_operator.h"

namespace MulticoreRuntime {

constexpr uint32_t L1_DUMP_MAGIC = 0x4C314450;
constexpr uint32_t L1_DUMP_VERSION = 1;
constexpr uint32_t L1_DUMP_MAX_RECORDS = 4096;
constexpr uint32_t L1_DUMP_PHASE_BEFORE = 1;
constexpr uint32_t L1_DUMP_PHASE_AFTER = 2;
constexpr uint32_t L1_DUMP_STATUS_PENDING = 0;
constexpr uint32_t L1_DUMP_STATUS_CAPTURED = 1;

struct L1DumpHeader {
  uint32_t magic;
  uint32_t version;
  uint32_t enabled;
  uint32_t record_count;
  uint32_t record_stride;
  uint32_t records_offset;
  uint32_t payload_offset;
  uint32_t total_bytes;
};

struct L1DumpRecord {
  uint32_t task_id;
  uint32_t phase;
  uint32_t parameter_position;
  uint32_t source_offset_low;
  uint32_t source_offset_high;
  uint32_t payload_offset;
  uint32_t payload_bytes;
  uint32_t status;
  uint32_t task_index;
  uint32_t reserved;
};

static_assert(sizeof(L1DumpHeader) == 32, "L1DumpHeader ABI must remain 32 bytes");
static_assert(sizeof(L1DumpRecord) == 40, "L1DumpRecord ABI must remain 40 bytes");

}  // namespace MulticoreRuntime

#endif  // MULTICORE_SCHEDULER_L1_TENSOR_DUMP_H
