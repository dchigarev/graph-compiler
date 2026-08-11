################################################################################
# Copyright (C) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
# SPDX-License-Identifier: Apache-2.0
################################################################################
#
# Sourced automatically in every CI step via BASH_ENV (see ci.yml env block).

# Saves failed test output to a dump file named after the test id and prints
# the dump file path to stdout.
save_test_dump() {
  local dump_dir="$1" test_id="$2" output="$3"
  local dump_file="$dump_dir/$(echo "$test_id" | sha1sum | cut -d' ' -f1).txt"
  mkdir -p "$dump_dir"
  echo "Test $test_id failed. Saving output to $dump_file" >&2
  echo "$output" > "$dump_file"
  echo "$dump_file"
}

# Reruns a test command with IR/debug dumping enabled, appending the output
# to the given dump file.
rerun_with_debug_dump() {
  local dump_file="$1"
  shift
  echo "Running test again with IR dumps enabled:" >> "$dump_file"
  echo "GC_DUMP_IR=1 OV_MLIR_DEBUG=1 GC_FILE_CACHE_OFF=1 $* &>> $dump_file" >> "$dump_file"
  GC_DUMP_IR=1 OV_MLIR_DEBUG=1 GC_FILE_CACHE_OFF=1 "$@" &>> "$dump_file"
}

export -f save_test_dump
export -f rerun_with_debug_dump
