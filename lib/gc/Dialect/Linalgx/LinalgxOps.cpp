//===-- LinalgxOps.cpp - linalgx dialect ops --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/Dialect/Linalgx/LinalgxOps.h"
#include "gc/Dialect/Linalgx/LinalgxDialect.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::linalgx;

//===----------------------------------------------------------------------===//
// AttentionOp
//===----------------------------------------------------------------------===//

void AttentionOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                        TypeRange results, Value query, Value key, Value value,
                        Value scale, Value output, ArrayAttr indexingMaps,
                        std::optional<Value> mask) {
  Value maskIn = mask.value_or(Value());
  build(odsBuilder, odsState, results, query, key, value, scale, maskIn, output,
        indexingMaps, DictionaryAttr());
}

void AttentionOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                        TypeRange results, ValueRange inputOperands,
                        ValueRange initOperands, ArrayAttr indexingMaps) {
  assert(inputOperands.size() < 6);
  assert(initOperands.size() == 1);
  Value mask = inputOperands.size() > 4 ? inputOperands[4] : Value();
  build(odsBuilder, odsState, results, inputOperands[0], inputOperands[1],
        inputOperands[2], inputOperands[3], mask, initOperands[0], indexingMaps,
        DictionaryAttr());
}

LogicalResult AttentionOp::verify() {
  // TODO: verify the correctness of indexing maps and shapes of operands.
  return success();
}

ArrayRef<int64_t> AttentionOp::getShape(OpOperand *opOperand) {
  assert(opOperand->getOwner() == this->getOperation());
  Type t = opOperand->get().getType();
  // A VectorType is an elemental type, do not consider its rank for the
  // operand.
  if (isa<VectorType>(t))
    return {};
  if (auto shapedType = ::llvm::dyn_cast<ShapedType>(t)) {
    // Failsafe.
    assert((isa<MemRefType>(t) || isa<RankedTensorType>(t)) &&
           "expected a ranked tensor or memref in LinalgInterface::getRank");
    return shapedType.getShape();
  }
  return {};
}

MutableOperandRange AttentionOp::getDpsInitsMutable() {
  return MutableOperandRange(*this, /*numInputs=*/getMask() ? 5 : 4,
                             /*numInits=*/1);
}

SmallVector<AffineMap> AttentionOp::getIndexingMapsArray() {
  return SmallVector<AffineMap>(
      getIndexingMaps().getAsValueRange<AffineMapAttr>());
}

void AttentionOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                              MLIRContext *ctx) {
  // FIXME: add canonicalization patterns
  // patterns.insert<StaticizeLinalgExtOp<AttentionOp>>(ctx);
}

AffineMap AttentionOp::getMatchingIndexingMap(OpOperand *operand) {
  return *(getIndexingMaps().getAsValueRange<AffineMapAttr>().begin() +
           operand->getOperandNumber());
}

//===----------------------------------------------------------------------===//
// GroupedMatmulOp
//===----------------------------------------------------------------------===//

LogicalResult GroupedMatmulOp::verify() {
  auto lhsTy = dyn_cast<ShapedType>(getLhs().getType());
  auto weiTy = dyn_cast<ShapedType>(getWeights().getType());
  auto idxTy = dyn_cast<ShapedType>(getIndices().getType());
  auto outTy = dyn_cast<ShapedType>(getOutput().getType());
  if (!lhsTy || !weiTy || !idxTy || !outTy)
    return emitOpError("expects shaped operands");
  if (lhsTy.getRank() != 3 || weiTy.getRank() != 3 || outTy.getRank() != 3)
    return emitOpError("expects rank-3 lhs [S, M, K], weights [E, N, K] and "
                       "output [S, M, N]");
  if (idxTy.getRank() != 2)
    return emitOpError("expects rank-2 indices [M, S]");
  if (!idxTy.getElementType().isSignlessInteger())
    return emitOpError("expects integer indices");
  if (getIndexingMaps().size() != 4)
    return emitOpError("expects 4 indexing maps (lhs, weights, indices, "
                       "output)");
  if (getTileM() <= 0)
    return emitOpError("expects a positive 'tile_m'");
  return success();
}

ArrayRef<int64_t> GroupedMatmulOp::getShape(OpOperand *opOperand) {
  assert(opOperand->getOwner() == this->getOperation());
  if (auto shapedType = dyn_cast<ShapedType>(opOperand->get().getType()))
    return shapedType.getShape();
  return {};
}

MutableOperandRange GroupedMatmulOp::getDpsInitsMutable() {
  return MutableOperandRange(*this, /*numInputs=*/3, /*numInits=*/1);
}

SmallVector<AffineMap> GroupedMatmulOp::getIndexingMapsArray() {
  return SmallVector<AffineMap>(
      getIndexingMaps().getAsValueRange<AffineMapAttr>());
}

AffineMap GroupedMatmulOp::getMatchingIndexingMap(OpOperand *operand) {
  return *(getIndexingMaps().getAsValueRange<AffineMapAttr>().begin() +
           operand->getOperandNumber());
}

/////// Operations corresponding to library calls defined with Tablegen ////////

#define GET_OP_CLASSES
#include "gc/Dialect/Linalgx/LinalgxOps.cpp.inc"
