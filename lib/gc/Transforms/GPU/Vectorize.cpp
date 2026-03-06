//===--------- Vectorize.cpp - Vectorize structured ops ----------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::gc {
#define GEN_PASS_DECL_VECTORIZE
#define GEN_PASS_DEF_VECTORIZE
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

struct VectorizationPattern : public RewritePattern {
  explicit VectorizationPattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rw) const override {
    if (!linalg::hasVectorizationImpl(op))
      return rw.notifyMatchFailure(op, "Unsupported Op, cannot vectorize");

    FailureOr<linalg::VectorizationResult> vectorResults =
        linalg::vectorize(rw, op, /*inputVectorSizes=*/{},
                          /*inputScalableVecDims=*/{});
    if (failed(vectorResults))
      return failure();

    rw.replaceOp(op, vectorResults->replacements);
    return success();
  }
};

/// Replaces broadcast + transpose with shape_cast + broadcast.
/// Example:
///   %0 = vector.broadcast %src : vector<128xf32> to vector<64x128xf32>
///   %1 = vector.transpose %0, [1, 0] : vector<64x128xf32> to vector<128x64xf32>
/// Becomes:
///   %0 = vector.shape_cast %src : vector<128xf32> to vector<128x1xf32>
///   %1 = vector.broadcast %0 : vector<128x1xf32> to vector<128x64xf32>
struct BroadcastTransposeToShapeCastBroadcast
    : public OpRewritePattern<vector::TransposeOp> {
  using OpRewritePattern<vector::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    // Check that the input to the transpose is a broadcast.
    auto broadcastOp =
        transposeOp.getVector().getDefiningOp<vector::BroadcastOp>();
    if (!broadcastOp)
      return failure();

    // Get the source of the broadcast.
    Value broadcastSrc = broadcastOp.getSource();
    auto srcType = dyn_cast<VectorType>(broadcastSrc.getType());
    if (!srcType)
      return failure();

    // We handle the case where source is 1D and broadcast result is 2D.
    auto broadcastResultType =
        cast<VectorType>(broadcastOp.getResult().getType());
    auto transposeResultType =
        cast<VectorType>(transposeOp.getResult().getType());

    if (srcType.getRank() != 1 || broadcastResultType.getRank() != 2)
      return failure();

    // Check the permutation is [1, 0].
    auto perm = transposeOp.getPermutation();
    if (perm.size() != 2 || perm[0] != 1 || perm[1] != 0)
      return failure();

    // Source shape is [N], broadcast to [M, N], transposed to [N, M].
    // Replace with: shape_cast [N] -> [N, 1], broadcast [N, 1] -> [N, M].
    int64_t N = srcType.getShape()[0];
    Type elemType = srcType.getElementType();

    auto shapeCastType = VectorType::get({N, 1}, elemType);
    Value shapeCast = rewriter.create<vector::ShapeCastOp>(
        transposeOp.getLoc(), shapeCastType, broadcastSrc);

    Value newBroadcast = rewriter.create<vector::BroadcastOp>(
        transposeOp.getLoc(), transposeResultType, shapeCast);

    rewriter.replaceOp(transposeOp, newBroadcast);
    return success();
  }
};

struct Vectorize final : gc::impl::VectorizeBase<Vectorize> {

  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    patterns.add<VectorizationPattern>(ctx);

    vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
    vector::populateVectorReductionToContractPatterns(patterns);
    vector::populateSinkVectorOpsPatterns(patterns);

    patterns.add<linalg::LinalgCopyVTRForwardingPattern,
                 linalg::LinalgCopyVTWForwardingPattern>(ctx, /*benefit=*/2);

    vector::TransferReadOp::getCanonicalizationPatterns(patterns, ctx);
    vector::TransferWriteOp::getCanonicalizationPatterns(patterns, ctx);
    tensor::populateFoldTensorSubsetIntoVectorTransferPatterns(patterns);

    patterns.add<linalg::CopyVectorizationPattern>(ctx);

    vector::populateFoldArithExtensionPatterns(patterns);

    linalg::populatePadOpVectorizationPatterns(patterns);
    linalg::populateDecomposePadPatterns(patterns);

    vector::populateVectorStepLoweringPatterns(patterns);

    patterns.add<BroadcastTransposeToShapeCastBroadcast>(ctx);

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns),
                                     GreedyRewriteConfig()))) {
      signalPassFailure();
    }
  }
};

} // namespace
