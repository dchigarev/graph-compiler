//===--------- RemoveAllocs.cpp - Remove unnecessary allocs --------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::gc {
#define GEN_PASS_DECL_REMOVEALLOCS
#define GEN_PASS_DEF_REMOVEALLOCS
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

static bool isConstantZero(Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value() == 0;
  return false;
}

static bool allZeroIndices(ValueRange idx) {
  for (Value v : idx)
    if (!isConstantZero(v))
      return false;
  return true;
}

/// Returns a permutation map suitable for transfer_write into a memref of rank M
/// from a vector of rank V.
///
/// - If M == V: identity.
/// - If M > V: project to the last V memref dims: (d0..dM-1) -> (d(M-V)..dM-1)
static AffineMap makeTrailingDimsPermutationMap(unsigned memrefRank,
                                                unsigned vectorRank,
                                                MLIRContext *ctx) {
  if (memrefRank == vectorRank)
    return AffineMap::getMultiDimIdentityMap(memrefRank, ctx);

  // Conservative: map vector dims to trailing memref dims.
  SmallVector<AffineExpr> exprs;
  exprs.reserve(vectorRank);
  for (unsigned i = memrefRank - vectorRank; i < memrefRank; ++i)
    exprs.push_back(getAffineDimExpr(i, ctx));
  return AffineMap::get(memrefRank, /*symbolCount=*/0, exprs, ctx);
}

/// A conservative "full transfer_write":
/// - no mask
/// - indices are all constant 0
static bool isFullTransferWriteAtZeros(vector::TransferWriteOp w) {
  if (w.getMask())
    return false;
  return allZeroIndices(w.getIndices());
}

struct FoldAllocTransferWriteExpandIntoCopy final
    : OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copy,
                                PatternRewriter &rewriter) const override {
    Value src = copy.getSource();
    Value dst = copy.getTarget();

    // Match optional expand_shape on the source.
    memref::ExpandShapeOp expand;
    if (auto ex = src.getDefiningOp<memref::ExpandShapeOp>()) {
      expand = ex;
      src = expand.getSrc();
    }

    // We want src to be an alloc.
    auto alloc = src.getDefiningOp<memref::AllocOp>();
    if (!alloc)
      return failure();

    // Find the unique transfer_write that fills this alloc.
    vector::TransferWriteOp fillWrite;
    for (Operation *user : alloc->getUsers()) {
      if (auto w = dyn_cast<vector::TransferWriteOp>(user)) {
        if (!isFullTransferWriteAtZeros(w))
          continue;
        // Must write INTO the alloc (or its exact SSA value).
        if (w.getBase() != alloc.getResult())
          continue;

        if (fillWrite)
          return failure(); // multiple writes => too risky
        fillWrite = w;
      }
    }
    if (!fillWrite)
      return failure();

    // The copy source must be either the alloc itself or expand_shape(alloc).
    // Also require that expand_shape is only used by this copy (to delete it safely).
    if (expand && !expand->hasOneUse())
      return failure();

    // We don't strictly require alloc one-use, but we will DCE later only if dead.

    // Vector to be written into the copy destination.
    Value vec = fillWrite.getVector();
    auto vecTy = dyn_cast<VectorType>(vec.getType());
    if (!vecTy)
      return failure();

    // Destination must be a memref.
    auto dstTy = dyn_cast<MemRefType>(dst.getType());
    if (!dstTy)
      return failure();

    unsigned dstRank = dstTy.getRank();
    unsigned vecRank = vecTy.getRank();
    if (dstRank < vecRank)
      return failure(); // can't map more vector dims than memref dims

    // Build zero indices for destination rank.
    Location loc = copy.getLoc();
    SmallVector<Value> zeroIdx;
    zeroIdx.reserve(dstRank);
    rewriter.setInsertionPoint(copy);
    for (unsigned i = 0; i < dstRank; ++i)
      zeroIdx.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));

    // Build in_bounds = [true ... true] for destination rank.
    SmallVector<bool> inBounds(dstRank, true);

// Build permutation_map: (d0..dM-1) -> (d(M-V)..dM-1)
    AffineMap permMap =
        makeTrailingDimsPermutationMap(dstRank, vecRank, rewriter.getContext());
    // permMap = AffineMap::getMultiDimIdentityMap(2, loc.getContext());
    // Create the new transfer_write into the copy destination.
    // This replaces: alloc + transfer_write + [expand_shape] + copy
// Build in_bounds as ArrayAttr<BoolAttr> of length == dstRank.
SmallVector<Attribute> inBoundsAttrs;
inBoundsAttrs.reserve(vecRank);
for (unsigned i = 0; i < vecRank; ++i)
  inBoundsAttrs.push_back(rewriter.getBoolAttr(true));
ArrayAttr inBoundsAttr = rewriter.getArrayAttr(inBoundsAttrs);
    vector::TransferWriteOp::create(
        rewriter,
        loc,
        vec,
        dst,
        zeroIdx,
        /*permutationMap=*/AffineMapAttr::get(permMap),
        /*mask=Value*/Value(),
        /*inBounds=*/inBoundsAttr);

    // Erase the copy.
    rewriter.eraseOp(copy);

    // Erase expand_shape if now dead.
    if (expand && expand->use_empty())
      rewriter.eraseOp(expand);

    // Erase the alloc + fillWrite if dead.
    if (fillWrite && fillWrite->use_empty())
      rewriter.eraseOp(fillWrite);

    if (alloc && alloc->use_empty())
      rewriter.eraseOp(alloc);

    return success();
  }
};

struct RemoveAllocs final : gc::impl::RemoveAllocsBase<RemoveAllocs> {
  void runOnOperation() override {
    func::FuncOp fn = getOperation();
    if (fn.isExternal())
      return;

    RewritePatternSet patterns(&getContext());
    patterns.add<FoldAllocTransferWriteExpandIntoCopy>(&getContext());

    GreedyRewriteConfig config;
    // Optimization-only: never fail the pass if patterns don't apply.
    (void)applyPatternsGreedily(fn, std::move(patterns), config);
  }
};

} // namespace