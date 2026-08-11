#!/bin/sh
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

set -e

# Default values
: ${GC_BUILD_TYPE:=RelWithDebInfo}
: ${GC_DYLINK:=OFF}

print_usage() {
    cat <<EOF
Usage:
$(basename "$0")
    [ -d | --dev     ] Development build
    [ -r | --release ] Release build (default: RelWithDebInfo)
    [ -l | --dyn     ] Dynamical linking, requires rebuild of LLVM, activates 'dev' option
    [ -c | --clean   ] Delete the build artifacts from the previous build
    [ -s | --suffix  ] Build dir suffix
    [ -h | --help    ] Print this message
    [ -v | --llvm    ] Build LLVM only
    [ -np | --no-patch ] Do not reset and apply patches to LLVM
EOF
}

for arg in "$@"; do
  case $arg in
    -d|--dev)
      GC_BUILD_TYPE="Debug"
      ;;
    -r|--release)
      GC_BUILD_TYPE="Release"
      ;;
    -c|--clean)
      CLEANUP=1
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    -l | --dyn)
      GC_DYLINK=ON
      ;;
    -np | --no-patch)
      LLVM_NO_PATCH=1
      ;;
    -v | --llvm)
      BUILD_LLVM_ONLY=1
      ;;
    *)
      echo "Unknown option: $arg"
      print_usage
      exit 1
      ;;
  esac
done

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
: ${GC_BUILD_DIR:="$PROJECT_DIR/build"}
: ${EXTERNALS_DIR:="$PROJECT_DIR/externals"}
: ${MAX_JOBS:=$(($(nproc) - 2))}
[ $MAX_JOBS -gt 0 ] || MAX_JOBS=2
GC_BUILD_DIR=$(realpath -m "$GC_BUILD_DIR")
EXTERNALS_DIR=$(realpath -m "$EXTERNALS_DIR")

repo_hash() {
  [ -d "$1" ] || return 0
  cd "$1"
  shift 1
  local rev=$(git rev-parse --short HEAD)
  local hash=$( (echo "$PROJECT_DIR/scripts/compile.sh" && \
    git ls-files --modified --others --exclude-standard && \
    git diff --name-only --cached) \
    | sort -u | xargs sha1sum)
  echo "${rev}-$( (echo "$hash" "$@") | sha1sum | cut -d' ' -f1)"
}


: ${LLVM_URL:='https://github.com/llvm/llvm-project.git'}
: ${LLVM_TAG:=$(cat "$PROJECT_DIR/cmake/llvm-version.txt")}
: ${LLVM_DIR:="$EXTERNALS_DIR/llvm-project"}
: ${LLVM_BUILD_DIR:="$LLVM_DIR/build"}
: ${LLVM_INSTALL_DIR:="$EXTERNALS_DIR/llvm"}
: ${LLVM_BUILD_TYPE:=$GC_BUILD_TYPE}
LLVM_DIR=$(realpath -m "$LLVM_DIR")
LLVM_BUILD_DIR=$(realpath -m "$LLVM_BUILD_DIR")
LLVM_INSTALL_DIR=$(realpath -m "$LLVM_INSTALL_DIR")

build_llvm() {
    local conf_files="$LLVM_BUILD_DIR/lib/cmake/llvm/LLVMConfig.cmake"

    if [ "$GITHUB_ACTIONS" = 'true' ]; then
      conf_files="$conf_files:$LLVM_INSTALL_DIR/lib/cmake/llvm/LLVMConfig.cmake"
      local build_target='--target install'
      MLIR_DIR="$LLVM_INSTALL_DIR/lib/cmake/mlir"
      if [ -f "$MLIR_DIR/MLIRConfig.cmake" ]; then
          echo "Using LLVM from CI cache $LLVM_INSTALL_DIR"
          return 0
      fi
    else
      MLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir"
      local build_target=''
    fi

    if ! [ -d "$LLVM_DIR" ]; then
        mkdir -p "$EXTERNALS_DIR"
        git init $LLVM_DIR
        cd "$LLVM_DIR"
        git remote add origin "$LLVM_URL"
        git fetch --no-tags --depth=1 origin ${LLVM_TAG}
        git checkout FETCH_HEAD
    elif [ -z "$LLVM_NO_PATCH" ]; then
        cd "$LLVM_DIR"
        [ $(git -C "$LLVM_DIR" rev-parse HEAD) = "$LLVM_TAG" ] || git fetch --no-tags --depth=1 origin ${LLVM_TAG}
        git reset --hard ${LLVM_TAG}
        [ -z "$CLEANUP" ] || git clean -xffd;
    else
      cd "$LLVM_DIR"
    fi

    if [ -z "$LLVM_NO_PATCH" ]; then
      for patch in "$PROJECT_DIR/patches/"*.patch; do
        if [ -f "$patch" ]; then
          echo "Applying patch: $patch"
          git apply --whitespace=fix "$patch"
        fi
      done
    fi

    LLVM_HASH=$(repo_hash "$LLVM_DIR" "$LLVM_INSTALL_DIR" $LLVM_BUILD_TYPE $GC_DYLINK)
    local hash_not_match=$(echo "$conf_files" | tr ':' '\n' | while read f; do
      grep -qF "set(LLVM_VERSION_SUFFIX $LLVM_HASH)" "$f" 2>/dev/null || echo 1; done)
    [ -z "$hash_not_match" ] && echo "LLVM build is up-to-date" && return 0

    [ -z "$CLEANUP" ] || rm -rf "$LLVM_BUILD_DIR"
    mkdir -p "$LLVM_BUILD_DIR"

    echo "Configuring LLVM..."
    cmake -G Ninja llvm -B "$LLVM_BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=$LLVM_BUILD_TYPE \
        -DCMAKE_CXX_FLAGS_DEBUG="-g -O0" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_ENABLE_RTTI=ON \
        -DLLVM_ENABLE_PROJECTS="mlir" \
        -DLLVM_TARGETS_TO_BUILD="X86" \
        -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD="SPIRV" \
        -DLLVM_BUILD_LLVM_DYLIB=$GC_DYLINK \
        -DLLVM_LINK_LLVM_DYLIB=$GC_DYLINK \
        -DLLVM_INCLUDE_RUNTIMES=OFF \
        -DLLVM_INCLUDE_EXAMPLES=OFF \
        -DLLVM_INCLUDE_TESTS=ON \
        -DLLVM_INCLUDE_BENCHMARKS=OFF \
        -DLLVM_INCLUDE_DOCS=OFF \
        -DLLVM_INSTALL_UTILS=ON \
        -DLLVM_INSTALL_GTEST=ON \
        -DLLVM_USE_PERF=ON \
        -DLLVM_ENABLE_BINDINGS=OFF \
        -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
        -DPython3_EXECUTABLE=$(which python3) \
        -DMLIR_ENABLE_LEVELZERO_RUNNER=OFF \
        -DLLVM_FORCE_VC_REPOSITORY="$LLVM_URL" \
        -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL_DIR"
    cmake --build "$LLVM_BUILD_DIR" --parallel $MAX_JOBS $build_target
    
    echo "$conf_files" | tr ':' '\n' | while read f; do
      [ -f "$f" ] && \
        sed -r -i "s/set *\(LLVM_VERSION_SUFFIX.+/set(LLVM_VERSION_SUFFIX $LLVM_HASH)/g" "$f"
    done
}

: ${GC_INSTALL_DIR:="$GC_BUILD_DIR/install"}
GC_INSTALL_DIR=$(realpath -m "$GC_INSTALL_DIR")

build_gc() {
  cd "$PROJECT_DIR"
  [ -z "$CLEANUP" ] || rm -rf "$GC_BUILD_DIR"

  local stamp_file="$GC_BUILD_DIR/gc_build_stamp.txt"
  mkdir -p "$GC_BUILD_DIR"
  touch "$stamp_file"
  local lit_path="$EXTERNALS_DIR/llvm-project/build/bin/llvm-lit"
  [ -f "$lit_path" ] || lit_path=$(which lit)
  local hash=$(repo_hash . "$MLIR_DIR" "$lit_path" $LLVM_HASH $GC_BUILD_TYPE $GC_DYLINK)
  local stamp=$(echo "$hash" && du -bs "$GC_BUILD_DIR" -X "$stamp_file")
  [ "$(cat "$stamp_file")" = "$stamp" ] && echo "GC build is up-to-date" && return 0
  
  cmake -S . --preset gc \
      -DCMAKE_BUILD_TYPE=$GC_BUILD_TYPE \
      -DGC_DYLINK=$GC_DYLINK \
      -DMLIR_DIR="$MLIR_DIR" \
      -DLLVM_EXTERNAL_LIT="$lit_path" \
      -DCMAKE_INSTALL_PREFIX="$GC_INSTALL_DIR"
  cmake --build "$GC_BUILD_DIR" --parallel $MAX_JOBS --target install

  stamp=$(echo "$hash" && du -bs "$GC_BUILD_DIR" -X "$stamp_file")
  echo "$stamp" > "$stamp_file"
}

echo "GC_BUILD_TYPE=$GC_BUILD_TYPE"
echo "GC_DYLINK=$GC_DYLINK"

build_llvm
[ -z "$BUILD_LLVM_ONLY" ] || exit 0
build_gc
