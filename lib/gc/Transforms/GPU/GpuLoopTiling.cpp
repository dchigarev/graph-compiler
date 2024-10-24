//===-- GpuLoopTiling.cpp - DESC --------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Transforms/RegionUtils.h"

#include "./GpuUtils.h"
#include "gc/Utils/Log.h"

using namespace mlir;
using namespace mlir::gc;

namespace mlir::gc {
#define GEN_PASS_DECL_GPULOOPTILING
#define GEN_PASS_DEF_GPULOOPTILING
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

struct GpuLoopTiling final : GpuPass<GpuLoopTiling>,
                             gc::impl::GpuLoopTilingBase<GpuLoopTiling> {
  friend GpuPass;
  explicit GpuLoopTiling() : GpuLoopTiling(gc::GpuLoopTilingOptions{}) {}
  explicit GpuLoopTiling(const gc::GpuLoopTilingOptions &opts)
      : GpuPass(), GpuLoopTilingBase(opts) {}

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    auto numThreads = getNumThreads(rewriter);
    getOperation().walk<WalkOrder::PreOrder>([&](scf::ParallelOp loop) {
      if (!loop->getParentOfType<scf::ParallelOp>()) {
        SmallVector<int64_t> tiles;
        auto steps = loop.getStep();
        tiles.reserve(steps.size());

        for (auto step : steps) {
          if (auto v = getConstIdxValue(step)) {
            tiles.push_back(v);
          } else {
            tiles.push_back(32);
          }
        }

        adjustTiles(numThreads, tiles);
        tileParallelLoop(loop, tiles, false);
      }
      return WalkResult::skip();
    });
    if (failed(simplifyRegions(rewriter, getOperation()->getRegions()))) {
      gcLogD("Failed to simplify regions");
    }
  }
};
} // namespace
