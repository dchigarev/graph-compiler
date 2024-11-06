//===- HackyMaskPass.cpp - A pass adding shared mem-space attr ----*- C++ -*-===//
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
#define GEN_PASS_DEF_HACKYMASKPASS
#include "gc/Transforms/Passes.h.inc"
} // namespace gc
} // namespace mlir

namespace {

bool isInGpuLaunch(Operation *op) {
  auto launchOp = op->getParentOfType<gpu::LaunchOp>();
  return launchOp != nullptr;
}

bool hasAssignedMemSpace(Value value) {
  if (auto memrefType = dyn_cast<MemRefType>(value.getType())) {
    if (memrefType.getMemorySpace()) {
      return true;
    }
  }
  return false;
}

bool isFunctionArg(Value value) {
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    return true;
  }
  return false;
}

template<typename XeOp>
struct InjectMask : public OpRewritePattern<XeOp> {
  using OpRewritePattern<XeOp>::OpRewritePattern;

  InjectMask(MLIRContext *ctx) : OpRewritePattern<XeOp>(ctx) {}

  LogicalResult matchAndRewrite(XeOp allocOp,
                                PatternRewriter &rewriter) const override {
    Location loc = allocOp.getLoc();
    if (isa<xegpu::LoadGatherOp>(allocOp)) {
      // allocOp.dump();
      auto x = allocOp.getOperand(1);
      if (!isFunctionArg(x)) {
        return success();
      }
      int64_t loadSize = 32;
      mlir::VectorType maskType = mlir::VectorType::get({loadSize}, rewriter.getI1Type());

      llvm::SmallVector<bool> maskValues;
      for (int i = 0; i < loadSize; i++) {
        maskValues.push_back(true);
      }
      mlir::DenseElementsAttr maskAttr = mlir::DenseIntElementsAttr::get(maskType, maskValues);

      // Create an arith.constant operation with the DenseElementsAttr
      auto mask = rewriter.create<mlir::arith::ConstantOp>(loc, maskType, maskAttr).getResult();
      x.replaceAllUsesWith(mask);
    } else {
      // allocOp.dump();
      // auto x = dyn_cast<xegpu::StoreScatterOp>(allocOp)->getOperand(3);
      // x.dump();
    }

    return success();
  }
};

struct HackyMaskPass : public gc::impl::HackyMaskPassBase<HackyMaskPass> {
  void runOnOperation() override {
    const auto ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<InjectMask<xegpu::LoadGatherOp>, InjectMask<xegpu::StoreScatterOp>>(patterns.getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace
