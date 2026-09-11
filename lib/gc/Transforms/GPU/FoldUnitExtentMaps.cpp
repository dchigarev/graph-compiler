//===- FoldUnitExtentMaps.cpp - Fold maps over unit-extent operands -*-C++-*-=//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::gc {
#define GEN_PASS_DECL_FOLDUNITEXTENTMAPS
#define GEN_PASS_DEF_FOLDUNITEXTENTMAPS
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

using namespace mlir;

namespace {

// An index map result that addresses a unit-extent operand dim can only ever
// evaluate to 0, so replace it with the constant. Dim and constant results are
// left alone: rewriting `d0` to `0` would drop the dim from the map and change
// the iteration domain, and constants are already folded.
struct FoldUnitExtentMapsPattern : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rw) const override {
    MLIRContext *ctx = op.getContext();
    AffineExpr zero = getAffineConstantExpr(0, ctx);
    SmallVector<AffineMap> maps = op.getIndexingMapsArray();
    bool changed = false;

    for (OpOperand &operand : op->getOpOperands()) {
      auto shapedTy = dyn_cast<ShapedType>(operand.get().getType());
      if (!shapedTy) continue;
      AffineMap &map = maps[operand.getOperandNumber()];
      if (map.getNumResults() != static_cast<unsigned>(shapedTy.getRank()))
        continue;

      SmallVector<AffineExpr> results(map.getResults());
      bool dirty = false;
      for (unsigned dim = 0, e = results.size(); dim < e; ++dim) {
        if (isa<AffineDimExpr, AffineConstantExpr>(results[dim])) continue;
        if (shapedTy.getDimSize(dim) != 1) continue;
        results[dim] = zero;
        dirty = true;
      }
      if (!dirty) continue;
      map = AffineMap::get(map.getNumDims(), map.getNumSymbols(), results, ctx);
      changed = true;
    }

    if (!changed) return failure();
    rw.modifyOpInPlace(
        op, [&]() { op.setIndexingMapsAttr(rw.getAffineMapArrayAttr(maps)); });
    return success();
  }
};

struct FoldUnitExtentMaps final
    : gc::impl::FoldUnitExtentMapsBase<FoldUnitExtentMaps> {

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldUnitExtentMapsPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
