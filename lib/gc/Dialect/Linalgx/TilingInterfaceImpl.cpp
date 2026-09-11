
#include "IndexingUtils.h"
#include "gc/Dialect/Linalgx/LinalgxDialect.h"
#include "gc/Dialect/Linalgx/LinalgxOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::linalgx;

namespace {

//===----------------------------------------------------------------------===//
// Attention Helpers
//===----------------------------------------------------------------------===//

Value getDimValue(OpBuilder &builder, Location loc, Value v, int64_t dim) {
  ShapedType type = cast<ShapedType>(v.getType());
  if (!type.isDynamicDim(dim)) {
    return arith::ConstantIndexOp::create(builder, loc, type.getDimSize(dim));
  }
  return TypeSwitch<Type, Value>(v.getType())
      .Case<RankedTensorType>([&](RankedTensorType t) -> Value {
        return builder.createOrFold<tensor::DimOp>(loc, v, dim);
      })
      .Case<MemRefType>([&](MemRefType t) -> Value {
        return builder.createOrFold<memref::DimOp>(loc, v, dim);
      });
}

OpFoldResult getDim(OpBuilder &builder, Location loc, Value v, int64_t dim) {
  auto t = cast<ShapedType>(v.getType());
  if (t.isDynamicDim(dim)) {
    return getDimValue(builder, loc, v, dim);
  }
  return builder.getIndexAttr(t.getDimSize(dim));
}

/// Permutes the offset and size arrays by the result indexes of the provided
/// affine map.
static SmallVector<Range> getPermutedRange(AffineMap permutation,
                                           ArrayRef<OpFoldResult> offsets,
                                           ArrayRef<OpFoldResult> sizes) {
  auto one = IntegerAttr::get(IndexType::get(permutation.getContext()), 1);
  assert(permutation.isProjectedPermutation() &&
         "Affine map should be a projected permutation");
  SmallVector<Range> output;
  for (AffineExpr dimExpr : permutation.getResults()) {
    int dim = cast<AffineDimExpr>(dimExpr).getPosition();
    output.push_back(Range{offsets[dim], sizes[dim], one});
  }
  return output;
}

Operation *getSlice(OpBuilder &b, Location loc, Value src,
                    ArrayRef<OpFoldResult> offsets,
                    ArrayRef<OpFoldResult> sizes,
                    ArrayRef<OpFoldResult> strides) {
  return TypeSwitch<Type, Operation *>(src.getType())
      .Case<RankedTensorType>([&](RankedTensorType t) -> Operation * {
        return tensor::ExtractSliceOp::create(b, loc, src, offsets, sizes,
                                              strides);
      })
      .Case<MemRefType>([&](MemRefType type) -> Operation * {
        return memref::SubViewOp::create(b, loc, src, offsets, sizes, strides);
      })
      .Default([&](Type t) -> Operation * {
        assert(false && "invalid type");
        return nullptr;
      });
}

Operation *getSlice(OpBuilder &b, Location loc, Value src,
                    ArrayRef<Range> slice) {
  SmallVector<OpFoldResult> offsets =
      llvm::map_to_vector(slice, [](Range x) { return x.offset; });
  SmallVector<OpFoldResult> sizes =
      llvm::map_to_vector(slice, [](Range x) { return x.size; });
  SmallVector<OpFoldResult> strides =
      llvm::map_to_vector(slice, [](Range x) { return x.stride; });
  return getSlice(b, loc, src, offsets, sizes, strides);
}

static SmallVector<Range>
getAttentionIterationDomain(Location loc, OpBuilder &b, int64_t domainRank,
                            ArrayRef<Value> values,
                            ArrayRef<AffineMap> indexingMaps) {
  SmallVector<Range> loopBounds(domainRank);
  OpFoldResult zero = b.getIndexAttr(0);
  OpFoldResult one = b.getIndexAttr(1);

  for (Range &bound : loopBounds) {
    bound.offset = zero;
    bound.stride = one;
  }

  SmallVector<bool> dimsFound(domainRank, false);
  auto fillSizes = [&](Value val, AffineMap indexingMap) {
    for (auto [idx, dimExpr] : llvm::enumerate(indexingMap.getResults())) {
      auto dim = cast<AffineDimExpr>(dimExpr);
      int64_t pos = dim.getPosition();
      if (dimsFound[pos]) {
        continue;
      }
      dimsFound[pos] = true;
      loopBounds[pos].size = getDim(b, loc, val, idx);
    }
  };

  for (auto [val, indexingMap] : llvm::zip_equal(values, indexingMaps)) {
    fillSizes(val, indexingMap);
  }

  return loopBounds;
}

static SmallVector<utils::IteratorType>
getAttentionIteratorTypes(int64_t domainRank, AffineMap qMap, AffineMap kMap,
                          AffineMap vMap, AffineMap oMap) {
  FailureOr<AttentionOpDetail> maybeOpInfo =
      AttentionOpDetail::get(qMap, kMap, vMap, oMap);
  assert(succeeded(maybeOpInfo) && "Failed to infer attention op details");
  AttentionOpDetail opInfo = maybeOpInfo.value();

  // All dimensions other than k1 and k2 are parallel.
  SmallVector<utils::IteratorType> iteratorTypes(domainRank,
                                                 utils::IteratorType::parallel);

  for (auto dim :
       llvm::concat<const int64_t>(opInfo.getK1Dims(), opInfo.getK2Dims())) {
    iteratorTypes[dim] = utils::IteratorType::reduction;
  }

  return iteratorTypes;
}

} // namespace

//===----------------------------------------------------------------------===//
// AttentionOp
//===----------------------------------------------------------------------===//

SmallVector<Range> AttentionOp::getIterationDomain(OpBuilder &b) {
  // Attention shape can be determined from Q, K, V alone.
  SmallVector<Value> shapedValues = {getQuery(), getKey(), getValue()};
  SmallVector<AffineMap> indexingMaps = {getQueryMap(), getKeyMap(),
                                         getValueMap()};
  return getAttentionIterationDomain(getLoc(), b, getIterationDomainRank(),
                                     shapedValues, indexingMaps);
}

SmallVector<utils::IteratorType> AttentionOp::getLoopIteratorTypes() {
  return getAttentionIteratorTypes(getIterationDomainRank(), getQueryMap(),
                                   getKeyMap(), getValueMap(), getOutputMap());
}

FailureOr<TilingResult>
AttentionOp::getTiledImplementation(OpBuilder &builder,
                                    ArrayRef<OpFoldResult> offsets,
                                    ArrayRef<OpFoldResult> sizes) {
  return getTiledImplementation(builder, offsets, sizes, {});
}

FailureOr<TilingResult> AttentionOp::getTiledImplementation(
    OpBuilder &builder, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes,
    ArrayRef<InnerTileAlignment> innerTileAlignments) {
  (void)innerTileAlignments;
  assert(offsets.size() == static_cast<size_t>(getIterationDomainRank()));
  assert(sizes.size() == static_cast<size_t>(getIterationDomainRank()));

  Location loc = getLoc();

  SmallVector<Range> querySlice =
      getPermutedRange(getQueryMap(), offsets, sizes);
  SmallVector<Range> keySlice = getPermutedRange(getKeyMap(), offsets, sizes);
  SmallVector<Range> valueSlice =
      getPermutedRange(getValueMap(), offsets, sizes);
  SmallVector<Range> outputSlice =
      getPermutedRange(getOutputMap(), offsets, sizes);

  Value scale = getScale();

  SmallVector<Value> tiledOperands;
  SmallVector<Operation *> slices;

  // Query
  {
    Operation *querySliceOp = getSlice(builder, loc, getQuery(), querySlice);
    tiledOperands.emplace_back(querySliceOp->getResult(0));
    slices.push_back(querySliceOp);
  }

  // Key
  {
    Operation *keySliceOp = getSlice(builder, loc, getKey(), keySlice);
    tiledOperands.emplace_back(keySliceOp->getResult(0));
    slices.push_back(keySliceOp);
  }

  // Value
  {
    Operation *valueSliceOp = getSlice(builder, loc, getValue(), valueSlice);
    tiledOperands.emplace_back(valueSliceOp->getResult(0));
    slices.push_back(valueSliceOp);
  }

  // Scale
  tiledOperands.emplace_back(scale);

  // Mask
  Value attnMask = getMask();
  if (attnMask) {
    SmallVector<Range> maskSlice =
        getPermutedRange(*getMaskMap(), offsets, sizes);
    Operation *maskSliceOp = getSlice(builder, loc, attnMask, maskSlice);
    tiledOperands.emplace_back(maskSliceOp->getResult(0));
    slices.push_back(maskSliceOp);
  }

  // Output
  {
    Operation *outputSliceOp = getSlice(builder, loc, getOutput(), outputSlice);
    tiledOperands.emplace_back(outputSliceOp->getResult(0));
    slices.push_back(outputSliceOp);
  }

  SmallVector<Type> resultTypes;
  if (hasPureTensorSemantics()) {
    int64_t baseIdx = attnMask ? 5 : 4;
    resultTypes.push_back(tiledOperands[baseIdx].getType());
  }

  Operation *tiledOp =
      mlir::clone(builder, getOperation(), resultTypes, tiledOperands);

  return TilingResult{
      {tiledOp}, SmallVector<Value>(tiledOp->getResults()), slices};
}

LogicalResult AttentionOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  resultOffsets.clear();
  resultSizes.clear();

  AffineMap resultIndexingMap;
  switch (resultNumber) {
  case 0:
    resultIndexingMap = getOutputMap();
    break;
  default:
    return failure();
  }

  for (AffineExpr dimExpr : resultIndexingMap.getResults()) {
    int dim = cast<AffineDimExpr>(dimExpr).getPosition();
    resultOffsets.push_back(offsets[dim]);
    resultSizes.push_back(sizes[dim]);
  }
  return success();
}

FailureOr<TilingResult>
AttentionOp::generateResultTileValue(OpBuilder &builder, unsigned resultNumber,
                                     ArrayRef<OpFoldResult> offsets,
                                     ArrayRef<OpFoldResult> sizes) {
  return generateResultTileValue(builder, resultNumber, offsets, sizes, {});
}

FailureOr<TilingResult> AttentionOp::generateResultTileValue(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes,
    ArrayRef<InnerTileAlignment> innerTileAlignments) {
  // Input offsets and sizes here are from the POV of the outputMap. We need to
  // normalize these offsets and size for it to be useful.

  // Initialize normalized offsets with 0s and normalized sizes with original
  // size.
  SmallVector<Range> iterationDomain(getIterationDomain(builder));
  SmallVector<OpFoldResult> normalizedSizes =
      llvm::map_to_vector(iterationDomain, [](Range x) { return x.size; });
  SmallVector<OpFoldResult> normalizedOffsets(getIterationDomainRank(),
                                              builder.getIndexAttr(0));
  ArrayRef<AffineExpr> outputDims = getOutputMap().getResults();
  for (size_t i = 0; i < outputDims.size(); i++) {
    int dim = cast<AffineDimExpr>(outputDims[i]).getPosition();
    normalizedOffsets[dim] = offsets[i];
    normalizedSizes[dim] = sizes[i];
  }
  return getTiledImplementation(builder, normalizedOffsets, normalizedSizes,
                                innerTileAlignments);
}

//===----------------------------------------------------------------------===//
// GroupedMatmulOp
//===----------------------------------------------------------------------===//

/// Position of `dim` in `map`'s results, or -1 if `map` does not address it.
static int64_t findDimInMap(AffineMap map, int64_t dim) {
  for (auto [i, expr] : llvm::enumerate(map.getResults()))
    if (auto d = dyn_cast<AffineDimExpr>(expr);
        d && static_cast<int64_t>(d.getPosition()) == dim)
      return static_cast<int64_t>(i);
  return -1;
}

SmallVector<Range> GroupedMatmulOp::getIterationDomain(OpBuilder &b) {
  Location loc = getLoc();
  OpFoldResult zero = b.getIndexAttr(0);
  OpFoldResult one = b.getIndexAttr(1);
  SmallVector<Range> bounds(4);
  for (Range &r : bounds) {
    r.offset = zero;
    r.stride = one;
  }
  // (slot, m, n) come from the output, k from the lhs. The k position inside
  // lhs is taken from the indexing map rather than hardcoded, so that the op
  // stays valid if lhs ever grows a leading dim again.
  bounds[kSlotDim].size = getDim(b, loc, getOutput(), 0);
  bounds[kMDim].size = getDim(b, loc, getOutput(), 1);
  bounds[kNDim].size = getDim(b, loc, getOutput(), 2);
  int64_t lhsK = findDimInMap(getLhsMap(), kKDim);
  assert(lhsK >= 0 && "lhs must address the reduction dim");
  bounds[kKDim].size = getDim(b, loc, getLhs(), lhsK);
  return bounds;
}

SmallVector<utils::IteratorType> GroupedMatmulOp::getLoopIteratorTypes() {
  return {utils::IteratorType::parallel, utils::IteratorType::parallel,
          utils::IteratorType::parallel, utils::IteratorType::reduction};
}

FailureOr<TilingResult>
GroupedMatmulOp::getTiledImplementation(OpBuilder &builder,
                                        ArrayRef<OpFoldResult> offsets,
                                        ArrayRef<OpFoldResult> sizes) {
  return getTiledImplementation(builder, offsets, sizes, {});
}

FailureOr<TilingResult> GroupedMatmulOp::getTiledImplementation(
    OpBuilder &builder, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes,
    ArrayRef<InnerTileAlignment> innerTileAlignments) {
  (void)innerTileAlignments;
  assert(offsets.size() == 4 && sizes.size() == 4);

  if (!hasPureTensorSemantics())
    return emitOpError("tiling is only implemented for tensor semantics");

  // The whole point of the op is that the expert index is uniform inside a
  // tile, which only holds if the slot dim is tiled to 1 (it is a leading dim,
  // so the tiling heuristics do that by default).
  auto slotTile = getConstantIntValue(sizes[kSlotDim]);
  if (!slotTile || *slotTile != 1)
    return emitOpError("expects the slot dimension to be tiled to 1");

  Location loc = getLoc();
  OpFoldResult one = builder.getIndexAttr(1);

  // Expert index of this tile: indices[m_offset, slot_offset].
  Value mIdx = getValueOrCreateConstantIndexOp(builder, loc, offsets[kMDim]);
  Value sIdx = getValueOrCreateConstantIndexOp(builder, loc, offsets[kSlotDim]);
  Value expertI =
      tensor::ExtractOp::create(builder, loc, getIndices(), ValueRange{mIdx, sIdx});
  Value expert = arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                            expertI);

  // Drops the leading dim of a [1, x, y] slice type, giving [x, y].
  auto dropLeadingDim = [](RankedTensorType t) {
    return RankedTensorType::get(t.getShape().drop_front(), t.getElementType());
  };

  // lhs [M, K] -> [tile_m, tile_k]: already 2-D, nothing to reduce.
  SmallVector<OpFoldResult> strides2(2, one);
  auto lhsSlice = tensor::ExtractSliceOp::create(
      builder, loc, getLhs(),
      SmallVector<OpFoldResult>{offsets[kMDim], offsets[kKDim]},
      SmallVector<OpFoldResult>{sizes[kMDim], sizes[kKDim]}, strides2);

  // weights [E, N, K] -> [tile_n, tile_k] and output [S, M, N] ->
  // [tile_m, tile_n]: the expert / slot dim is sliced to 1 and then dropped, so
  // the tile is 2-D. Rank-reducing extract_slice is handled by the tile-and-fuse
  // driver (tensor::replaceExtractSliceWithTiledProducer restores the reduced
  // type after fusing a producer), and the tiled value is written back with a
  // rank-reducing tensor.parallel_insert_slice, which is likewise legal.
  SmallVector<OpFoldResult> strides3(3, one);
  SmallVector<OpFoldResult> weiOffsets{expert, offsets[kNDim], offsets[kKDim]};
  SmallVector<OpFoldResult> weiSizes{one, sizes[kNDim], sizes[kKDim]};
  auto weiTileType = dropLeadingDim(tensor::ExtractSliceOp::inferResultType(
      cast<RankedTensorType>(getWeights().getType()), weiSizes));
  auto weiSlice = tensor::ExtractSliceOp::create(
      builder, loc, weiTileType, getWeights(), weiOffsets, weiSizes, strides3);

  SmallVector<OpFoldResult> outOffsets{offsets[kSlotDim], offsets[kMDim],
                                       offsets[kNDim]};
  SmallVector<OpFoldResult> outSizes{one, sizes[kMDim], sizes[kNDim]};
  auto outTileType = dropLeadingDim(tensor::ExtractSliceOp::inferResultType(
      cast<RankedTensorType>(getOutput().getType()), outSizes));
  auto outSlice = tensor::ExtractSliceOp::create(
      builder, loc, outTileType, getOutput(), outOffsets, outSizes, strides3);

  // With the expert resolved, the tile is a plain matmul with a transposed B:
  //   out[m, n] += lhs[m, k] * wei[n, k]
  MLIRContext *ctx = builder.getContext();
  AffineExpr m, n, k;
  bindDims(ctx, m, n, k);
  SmallVector<Attribute> maps = {
      AffineMapAttr::get(AffineMap::get(3, 0, {m, k}, ctx)),
      AffineMapAttr::get(AffineMap::get(3, 0, {n, k}, ctx)),
      AffineMapAttr::get(AffineMap::get(3, 0, {m, n}, ctx))};
  NamedAttribute indexingMaps(builder.getStringAttr("indexing_maps"),
                              builder.getArrayAttr(maps));

  auto matmul = linalg::MatmulOp::create(
      builder, loc, TypeRange{outTileType},
      ValueRange{lhsSlice.getResult(), weiSlice.getResult()},
      ValueRange{outSlice.getResult()}, ArrayRef<NamedAttribute>{indexingMaps});

  return TilingResult{{matmul.getOperation()},
                      SmallVector<Value>(matmul->getResults()),
                      {lhsSlice.getOperation(), weiSlice.getOperation(),
                       outSlice.getOperation()}};
}

LogicalResult GroupedMatmulOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  if (resultNumber != 0)
    return failure();
  resultOffsets.clear();
  resultSizes.clear();
  for (AffineExpr dimExpr : getOutputMap().getResults()) {
    int dim = cast<AffineDimExpr>(dimExpr).getPosition();
    resultOffsets.push_back(offsets[dim]);
    resultSizes.push_back(sizes[dim]);
  }
  return success();
}

FailureOr<TilingResult> GroupedMatmulOp::generateResultTileValue(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes) {
  return generateResultTileValue(builder, resultNumber, offsets, sizes, {});
}

FailureOr<TilingResult> GroupedMatmulOp::generateResultTileValue(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes,
    ArrayRef<InnerTileAlignment> innerTileAlignments) {
  SmallVector<Range> iterationDomain(getIterationDomain(builder));
  SmallVector<OpFoldResult> normalizedSizes =
      llvm::map_to_vector(iterationDomain, [](Range x) { return x.size; });
  SmallVector<OpFoldResult> normalizedOffsets(4, builder.getIndexAttr(0));
  ArrayRef<AffineExpr> outputDims = getOutputMap().getResults();
  for (size_t i = 0; i < outputDims.size(); i++) {
    int dim = cast<AffineDimExpr>(outputDims[i]).getPosition();
    normalizedOffsets[dim] = offsets[i];
    normalizedSizes[dim] = sizes[i];
  }
  auto tiled = getTiledImplementation(builder, normalizedOffsets,
                                      normalizedSizes, innerTileAlignments);
  if (failed(tiled))
    return failure();

  // getTiledImplementation() yields a 2-D tile, but a consumer that slices this
  // op's result expects a tile of the result's rank. Expand the dropped slot dim
  // back; tensor::replaceExtractSliceWithTiledProducer re-reduces it if the
  // consumer's slice was itself rank-reducing.
  assert(tiled->tiledValues.size() == 1);
  auto tileType = cast<RankedTensorType>(tiled->tiledValues[0].getType());
  auto expandedType = tensor::ExtractSliceOp::inferResultType(
      cast<RankedTensorType>(getOutput().getType()),
      SmallVector<OpFoldResult>{builder.getIndexAttr(1), normalizedSizes[kMDim],
                                normalizedSizes[kNDim]});
  if (tileType != expandedType) {
    auto reassociation =
        getReassociationIndicesForReshape(tileType, expandedType);
    if (!reassociation)
      return emitOpError("cannot expand the tiled result back to the tile shape "
                         "expected by a consumer");
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfterValue(tiled->tiledValues[0]);
    tiled->tiledValues[0] = tensor::ExpandShapeOp::create(
        builder, getLoc(), expandedType, tiled->tiledValues[0], *reassociation);
  }
  return tiled;
}