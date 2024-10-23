//===- LinalgToXeGPU.cpp - Linalg To XeGPU Lowering -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Transforms/Passes.h"

#include "gc/Transforms/Utils/MatcherUtils.h"
#include "gc/Transforms/Utils/StructuredOpMatcher.h"
#include "gc/Transforms/Utils/ValueUtils.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/TransformOps/Utils.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/TypeSwitch.h"

#include <numeric>
#include <optional>

using namespace mlir;
using namespace mlir::gc;

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_ALLOCSTOSLM
#include "gc/Transforms/Passes.h.inc"
} // namespace gc
} // namespace mlir

namespace {

bool isInGpuLaunch(mlir::Operation *op) {
  // Traverse up through parent operations
  mlir::Operation *parentOp = op;
  while (parentOp) {
    // Check if the current parent is a gpu.launch operation
    if (llvm::isa<mlir::gpu::LaunchOp>(parentOp)) {
      return true;
    }
    // Move to the parent operation
    parentOp = parentOp->getParentOp();
  }
  // If we reached the top without finding a gpu.launch, return false
  return false;
}

struct ConvertAlloc : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;
  // Constrain conversion to the supported GEMM-like ops.

  ConvertAlloc(MLIRContext *ctx)
      : OpRewritePattern<memref::AllocOp>(ctx) {}

  LogicalResult matchAndRewrite(memref::AllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    if (utils::hasSharedMemSpace(allocOp->getResult(0))) {
      return rewriter.notifyMatchFailure(
          allocOp, "Only support allocs in shared memory space");
    }

    if (!isInGpuLaunch(allocOp)) {
      return rewriter.notifyMatchFailure(
          allocOp, "Only support allocs in GPU regions");
    }

    mlir::Value memref = allocOp->getResult(0);
    mlir::MemRefType originalMemRefType = memref.getType().cast<mlir::MemRefType>();

    IntegerAttr sharedAddressSpace = IntegerAttr::get(
        rewriter.getIntegerType(3), 3);

    // Create a new MemRefType with the desired address space
    mlir::MemRefType newMemRefType = mlir::MemRefType::get(
        originalMemRefType.getShape(),                      // Same shape
        originalMemRefType.getElementType(),                // Same element type
        originalMemRefType.getLayout(),                     // Same layout
        sharedAddressSpace                                     // New address space
    );

    mlir::Value newMemRef = rewriter.create<memref::AllocOp>(
        allocOp.getLoc(), newMemRefType, allocOp.getOperands());
    
    memref.replaceAllUsesWith(newMemRef);
    
    return success();

    // add shared mem space
    
    // if (!isSharedMemSpace(allocOp)) {
    //   return rewriter.notifyMatchFailure(
    //       allocOp, "Only support allocs in shared memory space");
    // }


  }

};

struct AllocsToSLM : public gc::impl::AllocsToSLMBase<AllocsToSLM> {
  // using AllocsToSLM::AllocsToSLM;

  void runOnOperation() override {
    const auto ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<ConvertAlloc>(patterns.getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }

};

} // namespace
