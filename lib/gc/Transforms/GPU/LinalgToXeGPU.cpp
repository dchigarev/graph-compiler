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
using namespace mlir::xegpu;

namespace mlir {
namespace gc {
#define GEN_PASS_DEF_LINALGTOXEGPU
#include "gc/Transforms/Passes.h.inc"
} // namespace gc
} // namespace mlir

namespace {

// Represents VNNI configuration for an operand.
struct VnniConfig {
  int vnniFactor;
  int vnniAxis;
};

int getIntFromEnv(const std::string &envVarName) {
    const char* envValue = std::getenv(envVarName.c_str());
    if (envValue == nullptr) {
        return 16;  // Default to false if the variable is not set
    }

    std::string valueStr(envValue);
    std::transform(valueStr.begin(), valueStr.end(), valueStr.begin(), ::tolower);

    return std::stoi(valueStr);
}

// Helper struct to keep track of tiles' position with respect to whole matrix.
struct TilesArray {
  TilesArray() = delete;
  TilesArray(int numRows, int numCols) {
    assert(((numRows > 0) && (numCols > 0)) && "Expected 2D array shape");
    for (int i = 0; i < numRows; i++) {
      tileMatrix.push_back(SmallVector<Value>{});
      for (int j = 0; j < numCols; j++)
        tileMatrix[i].push_back(Value{});
    }
  }
  ~TilesArray() = default;

  Value getTile(int row, int col) { return tileMatrix[row][col]; }

  void setTile(int row, int col, Value val) { tileMatrix[row][col] = val; }

  SmallVector<Value> toFlatVector() {
    SmallVector<Value> flatVector;
    // NOLINTBEGIN
    for (auto row : tileMatrix)
      flatVector.append(row);
    // NOLINTEND
    return flatVector;
  }

  SmallVector<SmallVector<Value>> tileMatrix;
};

static int chunkSize = 1;

static bool hasSharedMemSpace(mlir::Value memref) {
  // Ensure the value is of MemRefType
  auto memRefType = memref.getType().dyn_cast<mlir::MemRefType>();
  if (!memRefType)
    return false;
  auto memorySpaceAttr = memRefType.getMemorySpace();
  if (!memorySpaceAttr)
    return false;
  auto gpuAttr = memorySpaceAttr.dyn_cast<IntegerAttr>();
  if (!gpuAttr)
    return false;
  
  return gpuAttr.getValue() == 3;
}

static xegpu::TensorDescType getTensorDescType(llvm::ArrayRef<int64_t> shape,
                                               mlir::Type elementType,
                                               mlir::xegpu::MemorySpace memory_space = mlir::xegpu::MemorySpace::Global,
                                               mlir::Attribute sg_map = {(mlir::Attribute::ImplType *)nullptr}) {
  if (memory_space == mlir::xegpu::MemorySpace::SLM) {
    return xegpu::TensorDescType::get(shape, elementType, chunkSize, xegpu::MemorySpace::SLM, nullptr);
  }
  return xegpu::TensorDescType::get(shape, elementType, /*array_length*/ 1,
                                    /*boundary_check*/ true, memory_space, sg_map);
}

// Return DPAS tile sizes if the gemm-like operation fits DPAS hardware.
static bool isDPASCompatible(linalg::LinalgOp linalgOp, int kTile,
                             ArrayRef<int64_t> dpasTile) {
  if (!(isa<linalg::MatmulOp>(linalgOp) ||
        isa<linalg::BatchReduceMatmulOp>(linalgOp) ||
        isa<linalg::MatmulTransposeBOp>(linalgOp) ||
        isa<linalg::GenericOp>(linalgOp))) {
    return false;
  }

  // Expect MxNxK DPAS register block sizes.
  if (dpasTile.size() != 3)
    return false;

  // Only static shapes are supported.
  if (linalgOp.hasDynamicShape())
    return false;

  auto aType = cast<ShapedType>(linalgOp.getDpsInputs()[0].getType());
  auto bType = cast<ShapedType>(linalgOp.getDpsInputs()[1].getType());
  auto cType = cast<ShapedType>(linalgOp.getDpsInits()[0].getType());

  auto elemTypeA = aType.getElementType();
  auto elemTypeB = bType.getElementType();
  auto elemTypeC = cType.getElementType();

  // TODO: Add more DPAS combinations.
  bool isSupportedPrecision =
      (elemTypeA.isF16() && elemTypeB.isF16() && elemTypeC.isF16()) ||
      (elemTypeA.isF16() && elemTypeB.isF16() && elemTypeC.isF32());
  if (!isSupportedPrecision)
    return false;

  auto mDim = cType.getShape()[0];
  auto nDim = cType.getShape()[1];
  auto kDim = aType.getShape().back();

  // Validate workload sizes.
  // The computation dimensions must fit into the tiles.
  // Reduction dimension tile size has to be compatible
  // with the warp tile.
  int dpasTileM = dpasTile[0];
  int dpasTileN = dpasTile[1];
  int dpasTileK = dpasTile[2];
  // NOLINTBEGIN
  if ((mDim % dpasTileM != 0) || (nDim % dpasTileN != 0) ||
      (kDim % dpasTileK != 0) || (kTile % dpasTileK != 0)) {
    return false;
  }
  // NOLINTEND

  return true;
}

// Verify if linalg operands fulfill lowering constraints.
static LogicalResult isValidMemrefOperand(linalg::LinalgOp linalgOp,
                                          Value operand,
                                          PatternRewriter &rewriter,
                                          unsigned maxDims = 2) {
  auto type = dyn_cast<MemRefType>(operand.getType());
  if (!type) {
    return rewriter.notifyMatchFailure(
        linalgOp, "Expect memref operand for XeGPU lowering");
  }

  if (type.getShape().size() > maxDims) {
    return rewriter.notifyMatchFailure(
        linalgOp, "Too high dimensionality for XeGPU operations");
  }

  auto strides = utils::getStaticStrides(operand);

  if (failed(strides)) {
    return rewriter.notifyMatchFailure(
        linalgOp, "Expect static strides for XeGPU lowering");
  }
  if (strides->back() != 1) {
    return rewriter.notifyMatchFailure(linalgOp,
                                       "Expect unit stride in the innermost "
                                       "dimension for XeGPU operations");
  }

  return success();
}

// Match and, if possible, lower a generic operation to an XeGPU compatible op.
// Returns the result of the lowered op or nullopt, otherwise.
static std::optional<Value> lowerGenericOp(linalg::GenericOp genericOp,
                                           ArrayRef<Value> operands,
                                           VectorType resType,
                                           PatternRewriter &rewriter) {
  Location loc = genericOp.getLoc();

  // Expect operands to be already loaded vectors.
  for (auto operand : operands) {
    if (!isa<VectorType>(operand.getType()))
      return std::nullopt;
  }

  if (structured_match::utils::isTwoDReluOp(genericOp,
                                            /*operands=*/nullptr)) { // NOLINT
    assert(operands.size() == 1 &&
           "Invalid number of operands for generic 2D ReLU");

    auto eltType = resType.getElementType();
    Value zeroConst;

    if (isa<FloatType>(eltType)) {
      auto floatType = cast<FloatType>(eltType);
      zeroConst = rewriter.create<arith::ConstantFloatOp>(
          loc, APFloat::getZero(floatType.getFloatSemantics()), floatType);
    } else if (isa<IntegerType>(eltType)) {
      zeroConst = rewriter.create<arith::ConstantIntOp>(loc, 0, eltType);
    } else {
      // Unhandled type. Bail out.
      return std::nullopt;
    }

    auto zeroVec =
        rewriter.create<vector::BroadcastOp>(loc, resType, zeroConst);

    return rewriter
        .create<arith::MaximumFOp>(loc, resType, operands[0], zeroVec)
        .getResult();
  }

  if (structured_match::utils::isTwoDAddOp(genericOp,
                                           /*operands=*/nullptr)) { // NOLINT
    assert(operands.size() == 2 &&
           "Invalid number of operands for generic 2D add");
    return rewriter
        .create<arith::AddFOp>(loc, resType, operands[0], operands[1])
        .getResult();
  }

  return std::nullopt;
}

// Lower an elementwise operation to an XeGPU compatible op.
// Returns the result of the lowered op or nullopt, otherwise.
static std::optional<Value> lowerEltwiseOp(linalg::LinalgOp linalgOp,
                                           ArrayRef<Value> operands,
                                           PatternRewriter &rewriter) {
  Location loc = linalgOp.getLoc();

  assert(llvm::all_of(operands,
                      [&](Value tile) {
                        return tile.getType() == operands[0].getType();
                      }) &&
         "All eltwise operands must have the same type.");

  // Expect operands to be already loaded vectors.
  for (auto operand : operands) {
    if (!isa<VectorType>(operand.getType()))
      return std::nullopt;
  }

  auto operandType = cast<ShapedType>(operands[0].getType());
  auto resType =
      VectorType::get(operandType.getShape(), operandType.getElementType());
  auto eltType = resType.getElementType();

  return llvm::TypeSwitch<Operation *, std::optional<Value>>(linalgOp)
      .Case([&](linalg::AbsOp absOp) -> std::optional<Value> {
        assert(operands.size() == 1 && "Invalid number of operands for abs");
        if (isa<FloatType>(eltType)) {
          return rewriter.create<math::AbsFOp>(loc, resType, operands[0])
              .getResult();
        }
        if (isa<IntegerType>(eltType)) {
          return rewriter.create<math::AbsIOp>(loc, resType, operands[0])
              .getResult();
        }
        // Unhandled type. Bail out.
        return std::nullopt;
      })
      .Case([&](linalg::AddOp addOp) -> std::optional<Value> {
        assert(operands.size() == 2 && "Invalid number of operands for add");
        if (isa<FloatType>(eltType)) {
          return rewriter
              .create<arith::AddFOp>(loc, resType, operands[0], operands[1])
              .getResult();
        }
        if (isa<IntegerType>(eltType)) {
          return rewriter
              .create<arith::AddIOp>(loc, resType, operands[0], operands[1])
              .getResult();
        }
        // Unhandled type. Bail out.
        return std::nullopt;
      })
      .Case([&](linalg::CeilOp ceilOp) -> std::optional<Value> {
        assert(operands.size() == 1 && "Invalid number of operands for ceil");
        return rewriter.create<math::CeilOp>(loc, resType, operands[0])
            .getResult();
      })
      .Case([&](linalg::DivOp divOp) -> std::optional<Value> {
        assert(operands.size() == 2 && "Invalid number of operands for div");
        if (isa<FloatType>(eltType)) {
          return rewriter
              .create<arith::DivFOp>(loc, resType, operands[0], operands[1])
              .getResult();
        }
        if (isa<IntegerType>(eltType)) {
          return rewriter
              .create<arith::DivSIOp>(loc, resType, operands[0], operands[1])
              .getResult();
        }
        // Unhandled type. Bail out.
        return std::nullopt;
      })
      .Case([&](linalg::DivUnsignedOp divUnsignedOp) -> std::optional<Value> {
        assert(operands.size() == 2 &&
               "Invalid number of operands for unsigned div");
        if (isa<IntegerType>(eltType)) {
          return rewriter
              .create<arith::DivUIOp>(loc, resType, operands[0], operands[1])
              .getResult();
        }
        // Unhandled type. Bail out.
        return std::nullopt;
      })
      .Case([&](linalg::ExpOp expOp) -> std::optional<Value> {
        assert(operands.size() == 1 && "Invalid number of operands for exp");
        return rewriter.create<math::ExpOp>(loc, resType, operands[0])
            .getResult();
      })
      .Case([&](linalg::FloorOp floorOp) -> std::optional<Value> {
        assert(operands.size() == 1 && "Invalid number of operands for floor");
        return rewriter.create<math::FloorOp>(loc, resType, operands[0])
            .getResult();
      })
      .Case([&](linalg::MaxOp maxOp) -> std::optional<Value> {
        assert(operands.size() == 2 && "Invalid number of operands for max");
        if (isa<FloatType>(eltType)) {
          return rewriter
              .create<arith::MaximumFOp>(loc, resType, operands[0], operands[1])
              .getResult();
        }
        if (isa<IntegerType>(eltType)) {
          if (eltType.isUnsignedInteger()) {
            return rewriter
                .create<arith::MaxUIOp>(loc, resType, operands[0], operands[1])
                .getResult();
          } else {
            return rewriter
                .create<arith::MaxSIOp>(loc, resType, operands[0], operands[1])
                .getResult();
          }
        }
        // Unhandled type. Bail out.
        return std::nullopt;
      })
      .Case([&](linalg::MulOp mulOp) -> std::optional<Value> {
        assert(operands.size() == 2 && "Invalid number of operands for mul");
        if (isa<FloatType>(eltType)) {
          return rewriter
              .create<arith::MulFOp>(loc, resType, operands[0], operands[1])
              .getResult();
        }
        if (isa<IntegerType>(eltType)) {
          return rewriter
              .create<arith::MulIOp>(loc, resType, operands[0], operands[1])
              .getResult();
        }
        // Unhandled type. Bail out.
        return std::nullopt;
      })
      .Case([&](linalg::NegFOp negFOp) -> std::optional<Value> {
        assert(operands.size() == 1 && "Invalid number of operands for negf");
        return rewriter.create<arith::NegFOp>(loc, resType, operands[0])
            .getResult();
      })
      .Case([&](linalg::SubOp subOp) -> std::optional<Value> {
        assert(operands.size() == 2 && "Invalid number of operands for sub");
        if (isa<FloatType>(eltType)) {
          return rewriter
              .create<arith::SubFOp>(loc, resType, operands[0], operands[1])
              .getResult();
        }
        if (isa<IntegerType>(eltType)) {
          return rewriter
              .create<arith::SubIOp>(loc, resType, operands[0], operands[1])
              .getResult();
        }
        // Unhandled type. Bail out.
        return std::nullopt;
      })
      .Case([&](linalg::GenericOp genericOp) -> std::optional<Value> {
        return lowerGenericOp(genericOp, operands, resType, rewriter);
      })
      .Default(
          [&](Operation *op) -> std::optional<Value> { return std::nullopt; });
}

// Get static GPU block sizes represented by a surrounding operation
// like a kernel launch or parallel loop.
// Returns known block sizes if they are all static or failure, otherwise.
static FailureOr<SmallVector<int64_t>> getStaticBlockSizes(Operation *op) {
  if (!op)
    return failure();

  auto getConstVal = [&](Value val) -> std::optional<int64_t> {
    if (auto constOp = val.getDefiningOp<arith::ConstantIndexOp>()) {
      return constOp.value();
    }
    return std::nullopt;
  };

  if (auto launchOp = dyn_cast<gpu::LaunchOp>(op)) {
    auto sizeX = getConstVal(launchOp.getBlockSizeX());
    auto sizeY = getConstVal(launchOp.getBlockSizeY());
    auto sizeZ = getConstVal(launchOp.getBlockSizeZ());
    if (!sizeX || !sizeY || !sizeZ)
      return failure();

    return SmallVector<int64_t>{*sizeX, *sizeY, *sizeZ};
  }

  // TODO: Remove when the lowering only occurs within a gpu.launch op.
  //       Manually computing this is brittle and duplicated parallel
  //       loops to gpu conversion.
  if (auto blockLoop = dyn_cast<scf::ParallelOp>(op)) {
    auto gridLoop = blockLoop->getParentOfType<scf::ParallelOp>();

    // Blocks or number of threads are represented by the first parallel loop
    // nested within another parallel loop.
    //
    // Fail if there is no outer parallel loop or current loop is nested more
    // than once.
    if (!gridLoop || (gridLoop->getParentOfType<scf::ParallelOp>())) {
      return failure();
    }

    SmallVector<int64_t> blockSizes;
    for (auto [lb, ub, step] :
         llvm::zip_equal(blockLoop.getLowerBound(), blockLoop.getUpperBound(),
                         blockLoop.getStep())) {
      auto lbVal = getConstVal(lb);
      auto ubVal = getConstVal(ub);
      auto stepVal = getConstVal(step);
      if (!lbVal || !ubVal || !stepVal)
        return failure();

      int64_t blockSize = (*ubVal - *lbVal) / *stepVal;

      // There must be at least one subgroup created for each dimension.
      // Otherwise, bail out and let kernel outlining fail later.
      if (blockSize <= 0)
        return failure();
      blockSizes.push_back(blockSize);
    }

    // Too many dimensions, something went wrong. Bail out.
    if (blockSizes.size() > 3)
      return failure();

    return blockSizes;
  }

  return failure();
}

// Get linearized GPU thread ID.
static Value getGpuLinearThreadId(PatternRewriter &rewriter, Location loc) {
  SmallVector<Value, 3> threadIds;
  SmallVector<Value, 3> blockDims;

  for (auto dim : {gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z}) {
    threadIds.push_back(rewriter.create<gpu::ThreadIdOp>(loc, dim));
    blockDims.push_back(rewriter.create<gpu::BlockDimOp>(loc, dim));
  }

  // The default GPU indexing is modeled after CUDA:
  // linear index = (z * sizeY + y) * sizeX + x
  Value threadId =
      rewriter.create<arith::MulIOp>(loc, threadIds[2], blockDims[1]);
  threadId = rewriter.create<arith::AddIOp>(loc, threadId, threadIds[1]);
  threadId = rewriter.create<arith::MulIOp>(loc, threadId, blockDims[0]);
  threadId = rewriter.create<arith::AddIOp>(loc, threadId, threadIds[0]);

  return threadId;
}

// Create a GEMM input tile to be loaded by each subgroup in
// cooperative fashion.
// Optionally accepts batch IV for batched GEMM input loading.
// Returns failure if it is unable to split block/workgroup for
// prefetching.
static FailureOr<xegpu::CreateNdDescOp>
createGemmCoopPrefetchTile(PatternRewriter &rewriter, linalg::LinalgOp linalgOp,
                           unsigned inputPos, int64_t numThreads,
                           ArrayRef<int> blockTile, ArrayRef<int> threadTile,
                           int tileStep) {
  assert(inputPos <= 1 && "Can handle only GEMM inputs: mat A or mat B");
  Location loc = linalgOp.getLoc();

  Value src = linalgOp.getDpsInputs()[inputPos];

  // Get a top level view into the whole matrix not only the thread slice.
  if (auto subview = dyn_cast_or_null<memref::SubViewOp>(src.getDefiningOp())) {
    src = subview.getSource();
  }

  const int tileRows = inputPos == 0 ? blockTile[0] : tileStep;
  const int tileCols = inputPos == 0 ? tileStep : blockTile[1];

  const int numElements = tileRows * tileCols;
  const int elementsPerThread = numElements / numThreads;

  // Limit the maximum prefetching row length to avoid very wide tiles.
  //
  // Currently, the max row size is capped by the hardware max load width.
  //
  // TODO: Expose as a tunable parameter or add some heuristics.
  const int maxRowLength = 32;

  // Prioritize first loading contiguous elements (row lenght/number of
  // columns) only then gather any remaining elements to be loaded from
  // further rows.
  // Also, ensure that the prefetch tile stays within the tile bounds.
  //
  // Ideally, prefetch tile sizes should be derived from total number of
  // elements to be loaded, number of threads/workitems, and hardware load
  // size limits. Large prefetch tiles might need to be split into sub-tiles.
  const int numCols =
      std::min(std::min(elementsPerThread, tileCols), maxRowLength);
  const int numRows = elementsPerThread / numCols;

  // Bail on invalid prefetching tiles config.
  if (numRows == 0 ||
      ((numRows * numCols * numThreads) > (tileRows * tileCols)))
    return failure();

  auto srcType = cast<ShapedType>(src.getType());

  auto prefetchType =
      getTensorDescType({numRows, numCols}, srcType.getElementType());

  Value threadId = getGpuLinearThreadId(rewriter, loc);

  // TODO: Simplify block offsets.
  //       Prefetching tile should be derived from the matmul op operands and
  //       exposed as a subview.
  //
  // Add offset if there are multiple blocks in the current tile's non-reduction
  // dimension.
  Value blockOffset = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  if (blockTile[inputPos] / threadTile[inputPos] > 1) {
    Value blockSize =
        rewriter.create<arith::ConstantIndexOp>(loc, blockTile[inputPos]);

    // For matrix B, pick correct block dimension.
    // Block min X has to be used if there is no thread tiling in the rows
    // (dim X) but only in columns (dim Y).
    gpu::Dimension gpuDim = gpu::Dimension::x;
    if ((inputPos == 1) && (blockTile[0] / threadTile[0] > 1)) {
      gpuDim = gpu::Dimension::y;
    }
    Value blockId = rewriter.create<gpu::BlockIdOp>(loc, gpuDim);

    blockOffset = rewriter.create<arith::MulIOp>(loc, blockId, blockSize);
  }

  Value numColTiles =
      rewriter.create<arith::ConstantIndexOp>(loc, tileStep / numCols);
  if (inputPos == 1) {
    numColTiles =
        rewriter.create<arith::ConstantIndexOp>(loc, blockTile[1] / numCols);
  }
  Value tileRowOffset =
      rewriter.create<arith::DivUIOp>(loc, threadId, numColTiles);
  Value tileColOffset =
      rewriter.create<arith::RemUIOp>(loc, threadId, numColTiles);

  Value tileRowSize = rewriter.create<arith::ConstantIndexOp>(loc, numRows);
  Value tileColSize = rewriter.create<arith::ConstantIndexOp>(loc, numCols);
  Value eltRowOffset =
      rewriter.create<arith::MulIOp>(loc, tileRowOffset, tileRowSize);
  Value eltColOffset =
      rewriter.create<arith::MulIOp>(loc, tileColOffset, tileColSize);

  if (inputPos == 0) {
    eltRowOffset =
        rewriter.create<arith::AddIOp>(loc, eltRowOffset, blockOffset);
  } else {
    eltColOffset =
        rewriter.create<arith::AddIOp>(loc, eltColOffset, blockOffset);
  }

  SmallVector<mlir::OpFoldResult> prefetchOffsets{eltRowOffset, eltColOffset};

  return rewriter.create<xegpu::CreateNdDescOp>(
      loc, prefetchType, dyn_cast<TypedValue<MemRefType>>(src),
      prefetchOffsets);
}

// Insert prefetches for the given tensor descriptors.
static void prefetchTiles(PatternRewriter &rewriter, Location loc,
                          ValueRange prefetchTiles,
                          xegpu::CachePolicyAttr readCacheHint) {
  // Prefetch the next set of input tiles.
  for (auto tile : prefetchTiles) {
    rewriter.create<xegpu::PrefetchNdOp>(loc, tile,
                                         /*l1_hint=*/readCacheHint,
                                         /*l2_hint=*/readCacheHint,
                                         /*l3_hint=*/readCacheHint);
  }
}

// Update all tensor descriptors offsets with the fixed offsets.
static SmallVector<Value> updateTilesOffsets(PatternRewriter &rewriter,
                                             Location loc, ValueRange tiles,
                                             ArrayRef<int64_t> offsets) {
  SmallVector<Value> updatedTiles;
  // convert static offsets to dynamic because of this IMEX bug:
  // https://github.com/intel/mlir-extensions/issues/815
  std::vector<Value> dynOffsets;
  for (auto &x : offsets) {
    Value offset = rewriter.create<arith::ConstantIndexOp>(loc, x);
    dynOffsets.push_back(offset);
  }
  ValueRange newOffsets{dynOffsets};
  for (auto tile : tiles) {
    auto updatedTile = rewriter
                           .create<xegpu::UpdateNdOffsetOp>(
                               loc, tile.getType(), tile,
                               /*offsets=*/newOffsets,
                               SmallVector<int64_t>{ShapedType::kDynamic,
                                                    ShapedType::kDynamic})
                           .getResult();
    updatedTiles.push_back(updatedTile);
  }

  return updatedTiles;
}

// Split a source into a series of descriptor tiles.
//
// The descriptors collectively load a 2D shape at the specified offsets from
// the given source.
// The offsets and the load shape must stay within the source boundaries.
//
// The descriptor sub-tiles are ordered in row-major fashion with respect to the
// whole load tile.
static SmallVector<Value>
createNdDescriptorTiles(PatternRewriter &rewriter, Location loc, Value src,
                      ArrayRef<int64_t> loadShape,
                      ArrayRef<int64_t> loadOffsets, ArrayRef<int64_t> descTile,
                      int arrayLength = 1, bool transpose = false) {
  assert(arrayLength == 1 && "Array descriptors are not supported");

  auto type = cast<ShapedType>(src.getType());
  auto descType = getTensorDescType(descTile, type.getElementType());

  // Create the root descriptor.
  //
  // It is more efficient to create remainig descriptors by only updating its
  // offsets compared to creating separate descriptors.
  // The original tile is split into contiguous sub-tiles so, the first tile
  // can be used as an anchor.
  Value rootOffsetRow =
      rewriter.create<arith::ConstantIndexOp>(loc, loadOffsets[0]);
  Value rootOffsetCol =
      rewriter.create<arith::ConstantIndexOp>(loc, loadOffsets[1]);

  mlir::SmallVector<mlir::OpFoldResult> offsets{rootOffsetRow, rootOffsetCol};
  auto rootTile =
      rewriter
          .create<xegpu::CreateNdDescOp>(
              loc, descType, dyn_cast<TypedValue<MemRefType>>(src), offsets)
          .getResult();

  SmallVector<Value> tiles;
  for (int i = 0; i < loadShape[0]; i += descTile[0]) {
    // convert static offsets to dynamic because of this IMEX bug:
    // https://github.com/intel/mlir-extensions/issues/815
    Value newRowOffs = rewriter.create<arith::ConstantIndexOp>(loc, i);
    for (int j = 0; j < loadShape[1]; j += descTile[1] * arrayLength) {
      Value newColOffs = rewriter.create<arith::ConstantIndexOp>(loc, j);
      if (transpose) {
        std::swap(newRowOffs, newColOffs);
      }
      auto tile = rewriter
                      .create<xegpu::UpdateNdOffsetOp>(
                          loc, descType, rootTile,
                          /*offsets=*/ValueRange{newRowOffs, newColOffs},
                          SmallVector<int64_t>{ShapedType::kDynamic,
                                               ShapedType::kDynamic})
                      .getResult();
      tiles.push_back(tile);
    }
  }

  return tiles;
}

static Value createIndexVector(PatternRewriter &rewriter, Location loc,
                               ArrayRef<int64_t> values) {
  mlir::VectorType vectorType = mlir::VectorType::get({static_cast<int64_t>(values.size())}, rewriter.getIndexType());
  mlir::DenseElementsAttr denseAttr = mlir::DenseIntElementsAttr::get(vectorType, values);
  auto vector = rewriter.create<mlir::arith::ConstantOp>(loc, vectorType, denseAttr).getResult();
  return vector;
}

static Value createIndexConstant(PatternRewriter &rewriter, Location loc,
                                 int64_t value) {
  return rewriter.create<arith::ConstantIndexOp>(loc, value);
}

static SmallVector<Value>
createScatterDescriptorTiles(PatternRewriter &rewriter, Location loc, Value flatMemref,
                      ArrayRef<int64_t> loadShape2D, ArrayRef<int64_t> descTile2D,
                      ArrayRef<int64_t> memrefStrides, Value slmBlockOffset) {
  int64_t maxLoadSize = 32;
  
  assert(memrefStrides.size() == 2 && "Strides must be 2D");
  assert(memrefStrides[1] == 1 && "Only row-major strides are supported");
  assert(loadShape2D.size() == 2 && "Load shape must be 2D");
  assert(loadShape2D[0] * loadShape2D[1] % maxLoadSize == 0 && "Load shape must be divisible by max load size");
  assert(descTile2D.size() == 2 && "Descriptor tile must be 2D");
  assert(maxLoadSize % descTile2D[1] == 0 && "Descriptor tile must be divisible by max load size");

  int64_t numLoadsPerTile = descTile2D[0] * descTile2D[1] / maxLoadSize;
  int64_t rowsPerLoad = maxLoadSize / descTile2D[1];
  int64_t numColTiles = loadShape2D[1] / descTile2D[1];

  auto memrefType = dyn_cast<MemRefType>(flatMemref.getType());

  SmallVector<SmallVector<int64_t>> offsetsShiftValues;
  for (int colTile = 0; colTile < numColTiles; colTile++) {
    offsetsShiftValues.push_back(SmallVector<int64_t>());
    for (int i = 0; i < rowsPerLoad; i++) {
      int64_t offset = i * memrefStrides[0];
      for (int j = 0; j < maxLoadSize / rowsPerLoad; j++) {
        offsetsShiftValues[colTile].push_back(offset + j + colTile * descTile2D[1]);
      }
    }
  }

  int64_t skipPerLoad = memrefStrides[0] * rowsPerLoad;
  auto offsetPerLoad = createIndexVector(rewriter, loc, SmallVector<int64_t>(32, skipPerLoad));

  auto offsetVecType = VectorType::get({maxLoadSize}, rewriter.getIndexType());
  auto descType = getTensorDescType(
    {maxLoadSize}, memrefType.getElementType(), xegpu::MemorySpace::SLM,
    xegpu::ScatterTensorDescAttr::get(rewriter.getContext(), xegpu::MemorySpace::SLM, /*chunkSize=*/1));

  
  SmallVector<Value> tiles;
  for (int i = 0; i < numColTiles; i++) {
    SmallVector<Value> slmBlockOffsetValues(32, slmBlockOffset);
    auto slmBlockOffsetV = rewriter.create<vector::FromElementsOp>(loc, offsetVecType, slmBlockOffsetValues);
    auto offsetsShift = createIndexVector(rewriter, loc, offsetsShiftValues[i]);

    auto offsets0 = rewriter.create<arith::AddIOp>(loc, slmBlockOffsetV, offsetsShift);

    auto desc = rewriter.create<xegpu::CreateDescOp>(loc, descType, flatMemref, offsets0).getResult();
    tiles.push_back(desc);
    for (int j = maxLoadSize; j < loadShape2D[0] * loadShape2D[1] / numColTiles; j+=maxLoadSize) {
      auto newTile = rewriter.create<xegpu::UpdateOffsetOp>(loc, descType, tiles.back(), offsetPerLoad).getResult();
      tiles.push_back(newTile);
    }
  }

  SmallVector<Value> transposedTiles;
  int numRowTiles = tiles.size() / numColTiles;
  llvm::dbgs() << "tiles.size() " << tiles.size() << "\n";
  llvm::dbgs() << "numColTiles " << numColTiles << "\n";
  llvm::dbgs() << "numRowTiles " << numRowTiles << "\n";
  // for (int group = 0; group < numColTiles ; group++) {
  //   for (int k = 0; k < numRowTiles / numLoadsPerTile; k++) {
  //     for (int loadOffset = 0; loadOffset < numLoadsPerTile; loadOffset++) {
  //       transposedTiles.push_back(tiles[group * numRowTiles + k * numLoadsPerTile + loadOffset]);
  //       llvm::dbgs() << "IDX " << group * numRowTiles + k * numLoadsPerTile + loadOffset << "\n";
  //     }
  //   }
  // }

  for (int rowTile = 0; rowTile < numRowTiles; rowTile+=numLoadsPerTile) {
    for (int colTile = 0; colTile < numColTiles; colTile++) {
      for (int loadOffset = 0; loadOffset < numLoadsPerTile; loadOffset++) {
        int newIdx = rowTile + colTile * numRowTiles + loadOffset;
        transposedTiles.push_back(tiles[newIdx]);
        // llvm::dbgs() << "IDX " << newIdx << "\n";
      }
    }
  }

  // for (int i = 0; i < numRowTiles; i++) {
  //   for (int j = 0; j < numColTiles; j++) {
  //     transposedTiles.push_back(tiles[i * numColTiles + j]);
  //   }
  // }
  llvm::dbgs() << "transposed.size() " << transposedTiles.size() << "\n";
  return transposedTiles;


  // auto type = cast<ShapedType>(src.getType());
  // auto descType = getTensorDescType(
  //   descTile[0], type.getElementType(), xegpu::MemorySpace::SLM,
  //   xegpu::ScatterTensorDescAttr::get(rewriter.getContext(), xegpu::MemorySpace::SLM, chunkSize));

  // // Create the root descriptor.
  // //
  // // It is more efficient to create remainig descriptors by only updating its
  // // offsets compared to creating separate descriptors.
  // // The original tile is split into contiguous sub-tiles so, the first tile
  // // can be used as an anchor.
  // mlir::VectorType offsetType = mlir::VectorType::get({static_cast<int64_t>(loadOffsets.size())}, rewriter.getIndexType());
  // mlir::DenseElementsAttr denseAttr = mlir::DenseIntElementsAttr::get(offsetType, loadOffsets);

  // // Create an arith.constant operation with the DenseElementsAttr
  // arith::ConstantOp offset = rewriter.create<mlir::arith::ConstantOp>(loc, offsetType, denseAttr);

  // Value initToSet = initialOffset ? initialOffset : offset.getResult();

  // auto rootTile =
  //     rewriter
  //         .create<xegpu::CreateDescOp>(
  //             loc, descType, dyn_cast<TypedValue<MemRefType>>(src), initToSet)
  //         .getResult();

  // SmallVector<Value> tiles;
  // Value prevTile = rootTile;
  // tiles.push_back(prevTile);
  
  // mlir::DenseElementsAttr shiftDenseAttr = mlir::DenseIntElementsAttr::get(offsetType, static_cast<int64_t>(32));
  // auto shiftOffset = rewriter.create<mlir::arith::ConstantOp>(loc, offsetType, shiftDenseAttr).getResult();

  // for (int j = descTile[0], tileIdx = 0; j < loadShape[0] * loadShape[1]; j += descTile[0], tileIdx++) {
  //   prevTile = rewriter
  //                   .create<xegpu::UpdateOffsetOp>(
  //                       loc, prevTile.getType(), prevTile,
  //                       /*offsets=*/shiftOffset)
  //                   .getResult();
  //   tiles.push_back(prevTile);
  // }

  // return tiles;
}


static SmallVector<Value>
createDescriptorTiles(PatternRewriter &rewriter, Location loc, Value src,
                      ArrayRef<int64_t> loadShape,
                      ArrayRef<int64_t> loadOffsets, ArrayRef<int64_t> descTile,
                      int arrayLength = 1, bool transpose = false) {
  
  if (hasSharedMemSpace(src)) {
    assert(false && "Shared memory is not supported yet");
    // return createScatterDescriptorTiles(rewriter, loc, src, loadShape, loadOffsets,
    //                                descTile, arrayLength, transpose);
  }
  return createNdDescriptorTiles(rewriter, loc, src, loadShape, loadOffsets,
                                 descTile, arrayLength, transpose);
}

SmallVector<int64_t> determine2DTileSize(ArrayRef<int64_t> totalShape, bool isVnni, int64_t elemByteWidth, int64_t rowTiles=-1) {
  // TODO: Fetch actual list of supported load configs.
  int64_t maxHeight = rowTiles == -1 ? 32 : rowTiles;
  int64_t maxWidth = 64 / elemByteWidth;
  // Assumes VNNI-factor 2.
  // TODO: Make the VNNI-factor flexible.
  if (isVnni)
    maxWidth /= 2;

  int64_t sgLoadRows = std::min(totalShape[0], maxHeight);
  int64_t sgLoadCols = std::min(totalShape[1], maxWidth);
  
  return SmallVector<int64_t>{sgLoadRows, sgLoadCols};
}

// Create coarse sub-tiles to be loaded by the current subgroup.
//
// The shape to be loaded is split into the largest 2D loads supported
// by the hardware.
//
// The load subgroup tiles are ordered in row-major fashion with respect to the
// source shape.
static SmallVector<Value> createCoarseNdDscTiles(PatternRewriter &rewriter,
                                               Location loc, Value src,
                                               ArrayRef<int64_t> sgTile,
                                               bool isVnni,
                                               bool transpose = false, int64_t rowTiles=-1) {
  assert(sgTile.size() <= 2 &&
         "Require at most 2D tile size for eltwise lowering");

  // Ensure that load is 2D.
  // TODO: Add support for 1D loads.
  SmallVector<int64_t, 2> sgTile2D{sgTile};
  if (sgTile.size() == 1)
    sgTile2D.push_back(1);

  auto type = cast<ShapedType>(src.getType());
  auto elemByteWidth = type.getElementType().getIntOrFloatBitWidth() / 8;
  auto tileSize = determine2DTileSize(sgTile, isVnni, elemByteWidth, rowTiles);

  // TODO: Add variable array_length support.
  int64_t arrayLength = 1;
  // NOLINTEND

  return createNdDescriptorTiles(rewriter, loc, src, sgTile2D, {0, 0},
                               tileSize, arrayLength,
                               transpose);
}

static Value flattenMemref(PatternRewriter &rewriter, Location loc, Value srcMemref) {
  auto srcType = cast<MemRefType>(srcMemref.getType());

  assert(srcType && "Expected a memref type");
  assert(srcType.getRank() == 2 && "Expected a 2D memref");
  
  int64_t flatSize = srcType.getShape()[0] * srcType.getShape()[1];

  Value offset = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value size = rewriter.create<arith::ConstantIndexOp>(loc, flatSize);
  Value stride = rewriter.create<arith::ConstantIndexOp>(loc, 1);

  // Use memref.reinterpret_cast to flatten the memref
  auto flatMemRefType = MemRefType::get({flatSize}, srcType.getElementType(), nullptr, srcType.getMemorySpace());
  auto flatMemref = rewriter.create<memref::ReinterpretCastOp>(
      loc, flatMemRefType, srcMemref, offset, size, stride).getResult();
  return flatMemref;
}


static SmallVector<Value> createSLMDescTiles(PatternRewriter &rewriter,
                                               Location loc, Value src,
                                               ArrayRef<int64_t> loadShape,
                                               ArrayRef<int64_t> descTile) {
  assert(loadShape.size() <= 2 &&
         "Require at most 2D tile size for eltwise lowering");
  
  auto srcTypeOrig = src.getType().cast<MemRefType>();
  assert(srcTypeOrig.getRank() == 2 && "Expected a 2D memref");
  auto elemByteWidth = srcTypeOrig.getElementType().getIntOrFloatBitWidth() / 8;
  
  // Get the shape of the original 2D memref
  ArrayRef<int64_t> srcShapeOrig = srcTypeOrig.getShape();

  // Flatten the 2D memref to 1D
  int64_t origFlatSize = srcShapeOrig[0] * srcShapeOrig[1];

  auto origSrc = src;
  Value initialOffsets;

  SmallVector<int64_t> memrefStrides;
  Value slmOffset;
  if (auto subView = dyn_cast<memref::SubViewOp>(src.getDefiningOp())) {
    auto xIntOffs = subView.getOffsets()[0];
    auto yIntOffs = subView.getOffsets()[1];

    // compute slm_offset (begining of the subview block in the original flat memref)
    auto slmBigWidthValue = cast<MemRefType>(subView.getOperand(0).getType()).getShape()[1];
    auto slmBigWidth = rewriter.create<arith::ConstantIndexOp>(loc, slmBigWidthValue);

    auto slmOffAdd0 = rewriter.create<arith::MulIOp>(loc, xIntOffs, slmBigWidth).getResult();
    slmOffset = rewriter.create<arith::AddIOp>(loc, slmOffAdd0, yIntOffs).getResult();

    memrefStrides = {slmBigWidthValue, 1};
    src = subView.getOperand(0);
  } else {
    // If the source is not a subview, then the slm_offset is 0
    slmOffset = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    memrefStrides = {srcShapeOrig[1], 1};
  }

  src = flattenMemref(rewriter, loc, src);

  return createScatterDescriptorTiles(rewriter, loc, /*flatMemref=*/src, /*loadShape2D=*/loadShape,
                               /*descTile2D=*/descTile, /*memrefStrides=*/memrefStrides,
                               /*slmBlockOffset=*/slmOffset);
}

static SmallVector<Value> createCoarseScatterDscTiles(PatternRewriter &rewriter,
                                               Location loc, Value src,
                                               ArrayRef<int64_t> loadShape,
                                               bool isVnni,
                                               bool transpose = false, int64_t rowTiles=-1) {
  assert(loadShape.size() <= 2 &&
         "Require at most 2D tile size for eltwise lowering");
  
  auto srcTypeOrig = src.getType().cast<MemRefType>();
  assert(srcTypeOrig.getRank() == 2 && "Expected a 2D memref");
  auto elemByteWidth = srcTypeOrig.getElementType().getIntOrFloatBitWidth() / 8;
  
  // Get the shape of the original 2D memref
  ArrayRef<int64_t> srcShapeOrig = srcTypeOrig.getShape();

  // Flatten the 2D memref to 1D
  int64_t origFlatSize = srcShapeOrig[0] * srcShapeOrig[1];

  auto origSrc = src;
  Value initialOffsets;

  int64_t maxHeight = rowTiles == -1 ? 32 : rowTiles;
  int64_t maxWidth = 64 / elemByteWidth;

  int64_t sgLoadRows = std::min(loadShape[0], maxHeight);
  int64_t sgLoadCols = std::min(loadShape[1], maxWidth);

  return createSLMDescTiles(rewriter, loc, /*flatMemref=*/src, /*loadShape2D=*/loadShape,
                               /*descTile2D=*/{sgLoadRows, sgLoadCols});
}

static SmallVector<Value> createCoarseDscTiles(PatternRewriter &rewriter,
                                               Location loc, Value src,
                                               ArrayRef<int64_t> sgTile,
                                               bool isVnni,
                                               bool transpose = false, int64_t rowTiles=-1) {
  if (hasSharedMemSpace(src)) {
    return createCoarseScatterDscTiles(rewriter, loc, src, sgTile, isVnni, transpose, rowTiles);
  }
  return createCoarseNdDscTiles(rewriter, loc, src, sgTile, isVnni, transpose, rowTiles);
}

// Return vector type with specified VNNI shape.
static VectorType getVnniVector(ArrayRef<int64_t> shape, Type elementType,
                                VnniConfig vnniConf) {
  assert(shape.size() == 2 && "Expected plain 2D shape");
  SmallVector<int64_t> vecShape{shape};
  vecShape[vnniConf.vnniAxis] /= vnniConf.vnniFactor;
  vecShape.push_back(vnniConf.vnniFactor);
  return VectorType::get(vecShape, elementType);
}

// Loads n-D tiles from memory to registers.
static SmallVector<Value>
loadNdDescTiles(PatternRewriter &rewriter, Location loc, ValueRange loadTiles,
                xegpu::CachePolicyAttr hint,
                std::optional<VnniConfig> vnniConf = std::nullopt,
                DenseI64ArrayAttr transpose = nullptr,
                IntegerAttr transpose_bit = nullptr) {
  // Assume all tiles have the same shape.
  auto tileType = cast<xegpu::TensorDescType>(loadTiles[0].getType());
  assert(llvm::all_of(loadTiles,
                      [&](Value tile) { return tile.getType() == tileType; }) &&
         "All load tiles must have the same type.");

  VectorType vecLoadType =
      VectorType::get(tileType.getShape(), tileType.getElementType());
  mlir::UnitAttr packedAttr = nullptr;
  if (vnniConf) {
    vecLoadType = getVnniVector(tileType.getShape(), tileType.getElementType(),
                                *vnniConf);
    if (!transpose_bit) {
      packedAttr = mlir::UnitAttr::get(rewriter.getContext());
    }
  }
  SmallVector<Value> loadVec;
  for (auto tile : loadTiles) {

    auto loadOp = rewriter.create<xegpu::LoadNdOp>(
        loc, vecLoadType, tile, packedAttr, transpose, transpose_bit,
        /*l1_hint=*/hint,
        /*l2_hint=*/hint, /*l3_hint=*/hint);
    loadVec.push_back(loadOp);
  }
  // TODO: Add split over the array_length > 1.
  //       The split must preserve row-major ordering of the load tiles.

  return loadVec;
}

static SmallVector<Value>
loadScatterDescTiles(PatternRewriter &rewriter, Location loc, ValueRange loadTiles,
                xegpu::CachePolicyAttr hint,
                std::optional<VnniConfig> vnniConf = std::nullopt,
                DenseI64ArrayAttr transpose = nullptr,
                IntegerAttr transpose_bit = nullptr,
                ArrayRef<int64_t> sgResShape = {}) {
  // Assume all tiles have the same shape.
  auto tileType = cast<xegpu::TensorDescType>(loadTiles[0].getType());
  assert(llvm::all_of(loadTiles,
                      [&](Value tile) { return tile.getType() == tileType; }) &&
         "All load tiles must have the same type.");

  VectorType vecLoadType =
      VectorType::get({tileType.getShape()[0]}, tileType.getElementType());
  mlir::UnitAttr packedAttr = nullptr;
  if (vnniConf) {
    assert(false && "VNNI not supported for scatter loads");
  }
  SmallVector<Value> normalizedVectors;

  int64_t loadSize = tileType.getShape()[0];
  mlir::VectorType maskVectorType = mlir::VectorType::get({loadSize}, rewriter.getI1Type());
  // mlir::VectorType maskVectorType2 = mlir::VectorType::get({loadSize}, rewriter.getI1Type());

  llvm::SmallVector<bool> maskValues;
  for (int i = 0; i < loadSize; i++) {
    maskValues.push_back(true);
  }
  mlir::DenseElementsAttr denseMaskAttr = mlir::DenseIntElementsAttr::get(maskVectorType, maskValues);
  auto mask = rewriter.create<mlir::arith::ConstantOp>(loadTiles[0].getLoc(), maskVectorType, denseMaskAttr);

  // mask = rewriter.create<vector::ShapeCastOp>(loc, maskVectorType2, mask);


  // int64_t loadCols;
  // int64_t loadRows;
  int64_t totalLoad;

  totalLoad = tileType.getShape()[0] * loadTiles.size();
  assert(totalLoad % 32 == 0 && "Total load size must be multiple of 32");
  int64_t elementsPerLoad = 32;

  // loadCols = 32;
  // loadRows = std::ceil(totalLoad / loadCols);

  int64_t chunkNRows = 32;
  int64_t chunkNCols = 32;
  if (sgResShape.size()) {
    chunkNRows = std::min(chunkNRows, sgResShape[0]);
    chunkNCols = std::min(chunkNCols, sgResShape[1]);
  }

  int64_t numChunks = std::max(std::ceil((totalLoad / elementsPerLoad) / chunkNRows), 1.0);
  int64_t totalChunkSize = totalLoad / numChunks;

  for (int64_t rowChunk = 0, loadTilesIdx=0; rowChunk < totalLoad / elementsPerLoad; rowChunk += chunkNRows) {
    int64_t totalLoadCh = std::min(totalChunkSize, totalLoad - rowChunk * elementsPerLoad);
    mlir::VectorType flatChunkType = mlir::VectorType::get({totalLoadCh}, tileType.getElementType());
    mlir::VectorType NdChunkType = mlir::VectorType::get({totalLoadCh / 32, 32}, tileType.getElementType());
    auto zeroAttr = DenseElementsAttr::get(flatChunkType, tileType.getElementTypeBitWidth() == 16 ? rewriter.getF16FloatAttr(0.0) : rewriter.getF32FloatAttr(0.0));
    Value tmp = rewriter.create<arith::ConstantOp>(loadTiles[0].getLoc(), flatChunkType, zeroAttr);

    Value loadAccumulator = rewriter.create<vector::ShapeCastOp>(loc, NdChunkType, tmp);

    for (int64_t i = 0; i < totalLoadCh / elementsPerLoad; i++) {
      auto tile = loadTiles[i + loadTilesIdx];
      auto loadOp = rewriter.create<xegpu::LoadGatherOp>(
          loc, vecLoadType, tile, /*mask=*/mask, nullptr// /*transpose=*/mlir::UnitAttr::get(rewriter.getContext())
          , nullptr, nullptr, nullptr
          );

      loadAccumulator = rewriter.create<vector::InsertOp>(
          loc, loadOp.getResult(), loadAccumulator, SmallVector<int64_t>{i});
    }
    loadTilesIdx += totalLoadCh / elementsPerLoad;

    if (sgResShape.size() == 0) {
      normalizedVectors.push_back(loadAccumulator);
      continue;
    }

    int64_t sgLoadRows = chunkNRows;
    int64_t sgLoadCols = chunkNCols;
    int64_t sgFlat = sgLoadRows * sgLoadCols;
    mlir::VectorType vectorTypeSg = mlir::VectorType::get({sgLoadRows, sgLoadCols}, tileType.getElementType());
    mlir::VectorType vectorTypeFlagSg = mlir::VectorType::get({sgFlat}, tileType.getElementType());

    auto flatLoaded = rewriter.create<vector::ShapeCastOp>(loc, flatChunkType, loadAccumulator);
    for (int64_t i = 0; i < totalLoadCh; i+= sgFlat) {
      auto slice = rewriter.create<vector::ExtractStridedSliceOp>(
            loc, flatLoaded, /*offsets=*/ArrayRef<int64_t>{i}, /*sizes=*/ArrayRef<int64_t>{sgFlat},
            /*strides=*/ArrayRef<int64_t>{1});
      auto res = rewriter.create<vector::ShapeCastOp>(loc, vectorTypeSg, slice);
      normalizedVectors.push_back(res);
    }
  }

  // verify correctness
  int64_t totalLoaded = 0;
  llvm::dbgs() << "Expected to load: " << totalLoad << "\n";
  for (auto v : normalizedVectors) {
    v.dump();
    auto shape = cast<VectorType>(v.getType()).getShape();
    totalLoaded += shape[0] * shape[1];
  }
  assert(totalLoaded == totalLoad);


  return normalizedVectors;
}

static SmallVector<Value>
loadDescTiles(PatternRewriter &rewriter, Location loc, ValueRange loadTiles,
                xegpu::CachePolicyAttr hint,
                std::optional<VnniConfig> vnniConf = std::nullopt,
                DenseI64ArrayAttr transpose = nullptr,
                IntegerAttr transpose_bit = nullptr,
                ArrayRef<int64_t> resultShape = {}) {
  auto tens = dyn_cast<xegpu::TensorDescType>(loadTiles[0].getType());
  if (tens.getMemorySpace() == xegpu::MemorySpace::SLM) {
    return loadScatterDescTiles(rewriter, loc, loadTiles, hint, vnniConf, transpose, transpose_bit, resultShape);
  }
  return loadNdDescTiles(rewriter, loc, loadTiles, hint, vnniConf, transpose, transpose_bit);
}

static SmallVector<Value>
storeNdDescTiles(PatternRewriter &rewriter, Location loc, SmallVector<Value> results, ValueRange loadTiles,
                xegpu::CachePolicyAttr hint) {
  SmallVector<Value> res;
  for (size_t i = 0; i < loadTiles.size(); i++) {
    auto val = rewriter.create<xegpu::StoreNdOp>(loc, results[i], loadTiles[i],
                                      /*l1_hint=*/hint,
                                      /*l2_hint=*/hint,
                                      /*l3_hint=*/hint);
    // res.push_back(val);
  }
  return res;
}

static SmallVector<Value>
storeScatterDescTiles(PatternRewriter &rewriter, Location loc, SmallVector<Value> results, ValueRange loadTiles,
                xegpu::CachePolicyAttr hint) {
  auto tileType = cast<xegpu::TensorDescType>(loadTiles[0].getType());
  assert(llvm::all_of(loadTiles,
                      [&](Value tile) { return tile.getType() == tileType; }) &&
         "All load tiles must have the same type.");

  VectorType vecStoreType =
      VectorType::get({tileType.getShape()[0]}, tileType.getElementType());


  SmallVector<Value> res;
  int64_t loadSize = tileType.getShape()[0];

  mlir::VectorType maskType = mlir::VectorType::get({loadSize}, rewriter.getI1Type());
  // mlir::VectorType maskType2 = mlir::VectorType::get({loadSize}, rewriter.getI1Type());

  llvm::SmallVector<bool> maskValues;
  for (int i = 0; i < loadSize; i++) {
    maskValues.push_back(true);
  }
  mlir::DenseElementsAttr maskAttr = mlir::DenseIntElementsAttr::get(maskType, maskValues);

  // Create an arith.constant operation with the DenseElementsAttr
  auto mask = rewriter.create<mlir::arith::ConstantOp>(loc, maskType, maskAttr);

  SmallVector<Value> chunkedResults;
  for (auto v : results) {
    auto resType = cast<VectorType>(v.getType());
    auto shape = resType.getShape();
    if (shape.size() == 1) {
      assert(shape[0] == 32);
      chunkedResults.push_back(v);
      continue;
    }
    if (shape[1] != 32) {
      v = rewriter.create<vector::ShapeCastOp>(loc, VectorType::get({(shape[0] * shape[1]) / 32, 32}, resType.getElementType()), v);
      shape = cast<VectorType>(v.getType()).getShape();
    }
    for (int i = 0; i < shape[0]; i++) {
      auto slice = rewriter.create<vector::ExtractOp>(loc, v, i);
      chunkedResults.push_back(slice);
    }
  }

  assert(chunkedResults.size() == loadTiles.size());

  for (size_t i = 0; i < loadTiles.size(); i++) {
    auto val = rewriter.create<xegpu::StoreScatterOp>(loc, chunkedResults[i], loadTiles[i],
                                      /*mask=*/mask,
                                      nullptr, // /*transpose=*/mlir::UnitAttr::get(rewriter.getContext()),
                                      nullptr, nullptr, nullptr);
  }
  return res;
}

static SmallVector<Value>
storeDescTiles(PatternRewriter &rewriter, Location loc, SmallVector<Value> results, ValueRange loadTiles,
                xegpu::CachePolicyAttr hint) {
  auto tens = dyn_cast<xegpu::TensorDescType>(loadTiles[0].getType());
  if (tens.getMemorySpace() == xegpu::MemorySpace::SLM) {
    return storeScatterDescTiles(rewriter, loc, results, loadTiles, hint);
  }
  return storeNdDescTiles(rewriter, loc, results, loadTiles, hint);
}

// Splits loaded tiles of a larger 2D tile into individual subtiles and places
// them in their corresponding positions with respect to the original large
// tile.
//
// The loaded tiles must be perfectly divisible by the specified subtiles.
// Assumes row-major ordering for both the loaded tiles and the original tile.
//
// If the loaded tiles use VNNI layout, corresponding VNNI configuration must be
// provided.
static TilesArray
extractVecSubTiles(PatternRewriter &rewriter, Location loc,
                   ValueRange loadVecTiles, ArrayRef<int64_t> sgTotalTile,
                   ArrayRef<int64_t> loadTile, ArrayRef<int64_t> subTile,
                   std::optional<VnniConfig> vnniConf = std::nullopt) {
  auto vecLoadType = cast<VectorType>(loadVecTiles[0].getType());
  assert(llvm::all_of(loadVecTiles,
                      [&](Value tile) {
                        return cast<VectorType>(tile.getType()) == vecLoadType;
                      }) &&
         "All loaded vectors must have the same type.");
  assert(vecLoadType.getShape().size() == 2 ||
         (vnniConf && "Requires VNNI config for non 2D loaded tiles"));

  // Accumulate all dimensions as the vector might have extra VNNI
  // dimensions.
  int loadVecSize = std::accumulate(vecLoadType.getShape().begin(),
                                    vecLoadType.getShape().end(), 1,
                                    std::multiplies<int64_t>());
  auto loadVecFlat = VectorType::get(loadVecSize, vecLoadType.getElementType());

  VectorType vecSubTileType =
      VectorType::get(subTile, vecLoadType.getElementType());
  if (vnniConf) {
    vecSubTileType =
        getVnniVector(subTile, vecLoadType.getElementType(), *vnniConf);
  }

  const int totalTileRows = sgTotalTile[0] / loadTile[0];
  const int totalTileCols = sgTotalTile[1] / loadTile[1];

  const int subTilesPerLoadRow = loadTile[0] / subTile[0];
  const int subTilePerLoadCol = loadTile[1] / subTile[1];

  const int subTileRows = sgTotalTile[0] / subTile[0];
  const int subTileCols = sgTotalTile[1] / subTile[1];
  TilesArray subTiles(subTileRows, subTileCols);

  // Iterate over the total tile.
  for (int m = 0; m < totalTileRows; m++) {
    for (int k = 0; k < totalTileCols; k++) {
      // Load tiles are ordered in row-major fashion.
      int loadIdx = m * totalTileCols + k;
      auto sgTotalTile = loadVecTiles[loadIdx];
      auto castFlat =
          rewriter.create<vector::ShapeCastOp>(loc, loadVecFlat, sgTotalTile);

      // Iterate over load tiles.
      // Each load tile contains one or more sub-tiles.
      for (int i = 0; i < subTilesPerLoadRow; i++) {
        for (int j = 0; j < subTilePerLoadCol; j++) {
          const int subTileSize = subTile[0] * subTile[1];
          int dpasIdx = i * subTilePerLoadCol + j;
          int offset = dpasIdx * subTileSize;

          auto slice = rewriter.create<vector::ExtractStridedSliceOp>(
              loc, castFlat, /*offsets=*/ArrayRef<int64_t>{offset},
              /*sizes=*/ArrayRef<int64_t>{subTileSize},
              /*strides=*/ArrayRef<int64_t>{1});
          auto castTile =
              rewriter.create<vector::ShapeCastOp>(loc, vecSubTileType, slice);

          // Insert the sub-tiles in their position relative to the whole
          // subgroup tile.
          int rowIdx = m * subTilesPerLoadRow + i;
          int colIdx = k * subTilePerLoadCol + j;
          subTiles.setTile(rowIdx, colIdx, castTile);
        }
      }
    }
  }

  return subTiles;
}

// Checks whether the given `matmulOperand` is produced by a
// `linalg::TransposeOp` and ensures that the transpose result is only used by
// valid operations, such as `linalg::MatmulOp`, `linalg::BatchReduceMatmulOp`,
// or `linalg::GenericOp`.
//
// If a valid transpose operation is found, the function records it for later
// removal and returns the operand of the transpose operation as the new matrix
// multiplication operand.
static FailureOr<Value> findAndReplaceTranspose(const Value &matmulOperand,
                                                size_t operandIdx,
                                                PatternRewriter &rewriter) {
  auto defOp = matmulOperand.getDefiningOp();
  if (!defOp) {
    return failure();
  }
  linalg::TransposeOp transposeOp = nullptr;

  for (auto x : defOp->getUsers()) {
    if (isa<linalg::TransposeOp>(x)) {
      if (transposeOp) {
        return rewriter.notifyMatchFailure(
            transposeOp, "Only one transpose operation is allowed");
      }

      transposeOp = dyn_cast<linalg::TransposeOp>(x);

      auto transposeRes = transposeOp.getDpsInits()[0];
      // verify that there are no other users of the transpose result
      // rather than our matmul
      for (auto trUser : transposeRes.getUsers()) {
        if (isa<linalg::MatmulOp>(trUser) ||
            isa<linalg::BatchReduceMatmulOp>(trUser) ||
            isa<linalg::GenericOp>(trUser)) {
          auto matmulOp = dyn_cast<linalg::LinalgOp>(trUser);
          auto actualMatmulOperand = matmulOp.getDpsInputs()[operandIdx];
          if (actualMatmulOperand != matmulOperand) {
            return rewriter.notifyMatchFailure(
                trUser,
                "Transpose result is used by more than one matmul operation");
          }
        } else if (isa<memref::DeallocOp>(trUser)) {
          // allow deallocs as users
          continue;
        } else if (isa<linalg::TransposeOp>(trUser)) {
          // check if it's the same transpose as we're processing
          if (!mlir::OperationEquivalence::isEquivalentTo(trUser, transposeOp,
                                                          /*flags=*/nullptr)) {
            return rewriter.notifyMatchFailure(
                trUser, "Only one transpose operation is allowed");
          }
          continue;
        } else {
          return rewriter.notifyMatchFailure(
              trUser,
              "Transpose result is not allowed to be used by this operation");
        }
      }
    }
  }
  if (transposeOp) {
    auto ret = transposeOp.getDpsInputs()[0];
    rewriter.eraseOp(transposeOp);
    return ret;
  }
  return rewriter.notifyMatchFailure(
      defOp, "No transpose operation producing the operand was found");
}

// Create XeGPU DPAS kernel out of GEMM-like operation.
static LogicalResult createDPASKernel(linalg::LinalgOp linalgOp,
                                      ArrayRef<int64_t> dpasTile, int kTile,
                                      int prefetchStages,
                                      PatternRewriter &rewriter) {
  assert((isa<linalg::MatmulOp>(linalgOp) ||
          isa<linalg::BatchReduceMatmulOp>(linalgOp) ||
          isa<linalg::MatmulTransposeBOp>(linalgOp) ||
          isa<linalg::GenericOp>(linalgOp)) &&
         "Requires a GEMM-like op for DPAS lowering");

  Location loc = linalgOp.getLoc();
  auto ctx = linalgOp.getContext();

  auto matA = linalgOp.getDpsInputs()[0];
  auto matB = linalgOp.getDpsInputs()[1];
  auto matC = linalgOp.getDpsInits()[0];

  bool transposeB = false;
  if (isa<linalg::MatmulTransposeBOp>(linalgOp)) {
    transposeB = true;
  } else {
    auto newMatB = findAndReplaceTranspose(matB, /*operandIdx=*/1, rewriter);
    if (!failed(newMatB)) {
      matB = *newMatB;
      transposeB = true;
    }
  }

  auto typeA = cast<ShapedType>(matA.getType());
  auto typeC = cast<ShapedType>(matC.getType());

  int64_t dpasTileM = dpasTile[0];
  int64_t dpasTileN = dpasTile[1];
  int64_t dpasTileK = dpasTile[2];

  // Cache hints for loads and stores.
  auto readCacheHint =
      xegpu::CachePolicyAttr::get(ctx, xegpu::CachePolicy::CACHED);
  auto writeCacheHint =
      xegpu::CachePolicyAttr::get(ctx, xegpu::CachePolicy::WRITE_BACK);

  bool isBrgemm = isa<linalg::BatchReduceMatmulOp>(linalgOp);

  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

  int dimM = typeC.getShape()[0];
  int dimN = typeC.getShape()[1];
  int dimK = typeA.getShape().back();

  // Create C sub-tiles.
  auto dpasTypeC =
      getTensorDescType({dpasTileM, dpasTileN}, typeC.getElementType());
  // SmallVector<Value> tilesC = createDescriptorTiles(
  //     rewriter, loc, matC, typeC.getShape(), {0, 0}, dpasTypeC.getShape());

  SmallVector<Value> tilesC;
  
  if (hasSharedMemSpace(matC))
    tilesC = createSLMDescTiles(rewriter, loc, matC, typeC.getShape(), dpasTypeC.getShape());
    // tilesC = createCoarseDscTiles(rewriter, loc, matC, typeC.getShape(), /*isVnni=*/true);
  else
    tilesC = createDescriptorTiles(
       rewriter, loc, matC, typeC.getShape(), {0, 0}, dpasTypeC.getShape());

  // Load C sub-tiles.
  // Fetch the inital values of the output accumulator.
  SmallVector<Value> loadVecC =
      loadDescTiles(rewriter, loc, tilesC, readCacheHint, std::nullopt, nullptr, nullptr, dpasTypeC.getShape());
  
  // llvm::dbgs() << "loadVecC.size(): " << loadVecC.size() << "\n";
  // for (auto v : loadVecC) {
  //   v.dump();
  // }

  // DPAS only works with F32 accumulators.
  auto dpasResType =
      VectorType::get(dpasTypeC.getShape(), FloatType::getF32(ctx));

  // Extend the accumulation values if needed.
  auto convOutPrecision = !typeC.getElementType().isF32();
  if (convOutPrecision) {
    for (size_t i = 0; i < loadVecC.size(); i++) {
      auto extOp =
          rewriter.create<arith::ExtFOp>(loc, dpasResType, loadVecC[i]);
      loadVecC[i] = extOp.getOut();
    }
  }

  // Create a loop and step into it.
  auto startLoop = [&](int lb, int ub, int step,
                       ValueRange iterArgs) -> scf::ForOp {
    Value lbCst = rewriter.create<arith::ConstantIndexOp>(loc, lb);
    Value ubCst = rewriter.create<arith::ConstantIndexOp>(loc, ub);
    Value stepCst = rewriter.create<arith::ConstantIndexOp>(loc, step);
    scf::ForOp loopOp =
        rewriter.create<scf::ForOp>(loc, lbCst, ubCst, stepCst, iterArgs);
    rewriter.setInsertionPointToStart(loopOp.getBody());
    return loopOp;
  };
  auto getLoopIterValues = [&](scf::ForOp loopOp) -> SmallVector<Value> {
    SmallVector<Value> loopIterVals;
    for (auto iterArg : loopOp.getRegionIterArgs())
      loopIterVals.push_back(iterArg);
    return loopIterVals;
  };

  OpBuilder::InsertionGuard guard(rewriter);

  // Construct and move into batch reduction loop.
  // Propagate output values as iter args.
  scf::ForOp batchLoop;
  Value batchIv;
  if (isBrgemm) {
    batchLoop = startLoop(0, typeA.getShape()[0], 1, loadVecC);
    batchIv = batchLoop.getInductionVar();
    loadVecC = getLoopIterValues(batchLoop);
    // TODO: Replace input matrices A and B with subviews on the current
    //       batchIV as loads can only be performed on 2D memrefs.
  }

  // Create A sub-tiles.
  SmallVector<Value> tilesA =
      createCoarseDscTiles(rewriter, loc, matA, {dimM, kTile}, /*isVnni=*/true);

  // Create B sub-tiles.
  SmallVector<Value> tilesB =
      createCoarseDscTiles(rewriter, loc, matB, {kTile, dimN},
                           /*isVnni=*/true, transposeB);

  // Create input prefetch tiles.
  int64_t numThreads = 1;
  auto blockDims =
      getStaticBlockSizes(linalgOp->getParentOfType<scf::ParallelOp>());
  if (succeeded(blockDims)) {
    numThreads = std::accumulate(blockDims->begin(), blockDims->end(), 1,
                                 std::multiplies<int64_t>());
  }
  // Disable prefetching when there is no block/workgroup parallelism.
  bool isCoopPrefetch = numThreads > 1;

  Value prefetchA;
  Value prefetchB;
  xegpu::TensorDescType prefetchTypeA;
  xegpu::TensorDescType prefetchTypeB;
  if (isCoopPrefetch) {
    // Return dimension size on which the whole block/workgroup operates.
    auto getBlockLevelSize = [&](Value val, int dim) -> int {
      if (auto subview =
              dyn_cast_or_null<memref::SubViewOp>(val.getDefiningOp())) {
        val = subview.getSource();
      }

      return cast<ShapedType>(val.getType()).getShape()[dim];
    };

    int blockRows = getBlockLevelSize(matC, 0);
    int blockCols = getBlockLevelSize(matC, 1);

    auto prefetchDescA = createGemmCoopPrefetchTile(
        rewriter, linalgOp, /*inputPos=*/0, numThreads, {blockRows, blockCols},
        {dimM, dimN}, kTile);
    auto prefetchDescB = createGemmCoopPrefetchTile(
        rewriter, linalgOp, /*inputPos=*/1, numThreads, {blockRows, blockCols},
        (transposeB) ? std::vector<int32_t>{dimM, dimN}
                     : std::vector<int32_t>{dimN, dimM},
        kTile);

    if (succeeded(prefetchDescA) && succeeded(prefetchDescB)) {
      prefetchA = prefetchDescA->getResult();
      prefetchTypeA = prefetchDescA->getType();
      prefetchB = prefetchDescB->getResult();
      prefetchTypeB = prefetchDescB->getType();

      // Start data prefetching by multistage data load.
      for (int i = 0; i < prefetchStages; i++) {
        prefetchTiles(rewriter, loc, ValueRange{prefetchA}, readCacheHint);
        prefetchTiles(rewriter, loc, ValueRange{prefetchB}, readCacheHint);
        prefetchA = updateTilesOffsets(rewriter, loc, ValueRange{prefetchA},
                                       {0, kTile})[0];
        prefetchB = updateTilesOffsets(rewriter, loc, ValueRange{prefetchB},
                                       (transposeB)
                                           ? std::vector<int64_t>{0, kTile}
                                           : std::vector<int64_t>{kTile, 0})[0];
      }
    } else {
      // Disable coop prefetching on failure.
      isCoopPrefetch = false;
    }
  }

  // Construct and move into GEMM reduction dimension tiling loop.
  // Propagate output values as iter args.
  SmallVector<Value> iterArgs;
  iterArgs.append(loadVecC);
  iterArgs.append(tilesA);
  iterArgs.append(tilesB);
  if (isCoopPrefetch) {
    iterArgs.push_back(prefetchA);
    iterArgs.push_back(prefetchB);
  }
  scf::ForOp kDimLoop = startLoop(0, dimK, kTile, iterArgs);
  auto iterValues = getLoopIterValues(kDimLoop);

  loadVecC = SmallVector<Value>{iterValues.begin(),
                                iterValues.begin() + loadVecC.size()};
  tilesA =
      SmallVector<Value>{iterValues.begin() + loadVecC.size(),
                         iterValues.begin() + loadVecC.size() + tilesA.size()};
  tilesB = SmallVector<Value>{
      iterValues.begin() + loadVecC.size() + tilesA.size(),
      iterValues.begin() + loadVecC.size() + tilesA.size() + tilesB.size()};
  if (isCoopPrefetch) {
    prefetchA = *(iterValues.end() - 2);
    prefetchB = *(iterValues.end() - 1);
  }

  // Periodically synchronize the block/workgroup to minimize impact on cache
  // due to replacement of sub-tiles before all threads/workitems consumed
  // inputs for reduction dimension step.
  //
  // TODO: Synchronization frequency should be derived from tile and cache size.
  int syncFreq = 4;
  int maxSyncStep = 1024;
  int syncStep = std::min(std::max(dimK / syncFreq, maxSyncStep), maxSyncStep);
  auto syncStepConst = rewriter.create<arith::ConstantIndexOp>(loc, syncStep);
  auto loopStepMod = rewriter.create<arith::RemUIOp>(
      loc, kDimLoop.getInductionVar(), syncStepConst);
  auto syncBlockCond = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, loopStepMod, zero);
  rewriter.create<scf::IfOp>(
      loc, syncBlockCond,
      /*thenBuilder=*/
      [](OpBuilder &b, Location loc) {
        b.create<gpu::BarrierOp>(loc);
        b.create<scf::YieldOp>(loc);
      },
      /*elseBuilder=*/nullptr);

  // TODO: Add more possible types.
  int vnniFactor = TypeSwitch<Type, int>(typeA.getElementType())
                       .Case([](Float16Type type) { return 2; })
                       .Default([](Type type) { return -1; });
  if (vnniFactor == -1)
    return failure();

  VnniConfig vnniConfB{.vnniFactor = vnniFactor, .vnniAxis = 0};

  // Load A sub-tiles.
  SmallVector<Value> loadVecA =
      loadNdDescTiles(rewriter, loc, tilesA, readCacheHint);
  auto tileTypeA = cast<xegpu::TensorDescType>(tilesA[0].getType());

  DenseI64ArrayAttr transpose = nullptr;
  IntegerAttr transpose_bit = nullptr;

  if (transposeB) {
    transpose_bit = rewriter.getIntegerAttr(rewriter.getIntegerType(32), 32);
    transpose = DenseI64ArrayAttr::get(rewriter.getContext(), {1, 0});
  }

  // Load B sub-tiles.
  SmallVector<Value> loadVecB =
      loadNdDescTiles(rewriter, loc, tilesB, readCacheHint, vnniConfB,
                      transpose, transpose_bit);
  auto tileTypeB = cast<xegpu::TensorDescType>(tilesB[0].getType());

  // Update offsets of the input tiles.
  // Shift along the reduction dimension.
  tilesA = updateTilesOffsets(rewriter, loc, tilesA, {0, kTile});
  tilesB = updateTilesOffsets(rewriter, loc, tilesB,
                              transposeB ? std::vector<int64_t>{0, kTile}
                                         : std::vector<int64_t>{kTile, 0});

  // Prefetch the next set of input tiles.
  if (isCoopPrefetch) {
    // Prefetch all block/workgroup tiles cooperatively.
    prefetchTiles(rewriter, loc, ValueRange{prefetchA}, readCacheHint);
    prefetchTiles(rewriter, loc, ValueRange{prefetchB}, readCacheHint);
    prefetchA =
        updateTilesOffsets(rewriter, loc, ValueRange{prefetchA}, {0, kTile})[0];
    prefetchB =
        updateTilesOffsets(rewriter, loc, ValueRange{prefetchB},
                           transposeB ? std::vector<int64_t>{0, kTile}
                                      : std::vector<int64_t>{kTile, 0})[0];
  } else {
    // Apply naive prefetching for each subgroup separately.
    prefetchTiles(rewriter, loc, tilesA, readCacheHint);
    prefetchTiles(rewriter, loc, tilesB, readCacheHint);
  }

  // Extract DPAS tiles from loaded sub-tiles.
  TilesArray dpasVecA =
      extractVecSubTiles(rewriter, loc, loadVecA, {dimM, kTile},
                         tileTypeA.getShape(), {dpasTileM, dpasTileK});
  TilesArray dpasVecB = extractVecSubTiles(rewriter, loc, loadVecB,
                                           {kTile, dimN}, tileTypeB.getShape(),
                                           {dpasTileK, dpasTileN}, vnniConfB);

  const int numTilesM = dimM / dpasTileM;
  const int numTilesN = dimN / dpasTileN;
  const int numTilesK = kTile / dpasTileK;

  // Compute sub-tiles of the C tile.
  //
  // Iterate over the reduction dimension sub-tiles as the outermost
  // loop to minimize read after write conflicts between partial
  // computations of the same C sub-tile.
  SmallVector<Value> dpasResults = loadVecC;

  for (int k = 0; k < numTilesK; k++) {
    for (int m = 0; m < numTilesM; m++) {
      for (int n = 0; n < numTilesN; n++) {
        int cIdx = m * numTilesN + n;

        Value result = rewriter
                           .create<xegpu::DpasOp>(
                               loc, dpasResType, dpasVecA.getTile(m, k),
                               dpasVecB.getTile(k, n), dpasResults[cIdx])
                           .getResult();

        // Update sub-tile partial result.
        dpasResults[cIdx] = result;
      }
    }
  }

  // Create loop terminator and exit the loop.
  auto terminateLoop = [&](scf::ForOp loopOp,
                           SmallVector<Value> resultValues) { // NOLINT
    rewriter.setInsertionPointToEnd(loopOp.getBody());
    rewriter.create<scf::YieldOp>(loc, resultValues);
    rewriter.setInsertionPointAfter(loopOp);
  };

  SmallVector<Value> yieldVals;
  yieldVals.append(dpasResults);
  yieldVals.append(tilesA);
  yieldVals.append(tilesB);
  if (isCoopPrefetch) {
    yieldVals.push_back(prefetchA);
    yieldVals.push_back(prefetchB);
  }

  // Terminate and exit reduction dim loop.
  terminateLoop(kDimLoop, yieldVals);
  yieldVals = kDimLoop.getResults();

  SmallVector<Value> results{yieldVals.begin(),
                             yieldVals.begin() + dpasResults.size()};

  // Terminate and exit batch reduce loop.
  if (isBrgemm) {
    terminateLoop(batchLoop, results);
    results = batchLoop.getResults();
  }

  // Truncate the result values if needed.
  if (convOutPrecision) {
    auto truncType =
        VectorType::get(dpasTypeC.getShape(), typeC.getElementType());
    for (size_t i = 0; i < results.size(); i++) {
      auto truncOp =
          rewriter.create<arith::TruncFOp>(loc, truncType, results[i]);
      results[i] = truncOp.getOut();
    }
  }

  // Write back the final C sub-tiles results to the output buffer.
  // SmallVector<xegpu::StoreNdOp> storeOps;
  // for (size_t i = 0; i < tilesC.size(); i++) {
  //   auto storeOp =
  //       rewriter.create<xegpu::StoreNdOp>(loc, results[i], tilesC[i],
  //                                         /*l1_hint=*/writeCacheHint,
  //                                         /*l2_hint=*/writeCacheHint,
  //                                         /*l3_hint=*/writeCacheHint);
  //   storeOps.push_back(storeOp);
  // }
  storeDescTiles(rewriter, loc, results, tilesC, writeCacheHint);

  rewriter.eraseOp(linalgOp);

  return success();
}

// Create XeGPU kernel out of elementwise operation.
LogicalResult createEltwiseKernel(linalg::LinalgOp linalgOp,
                                  PatternRewriter &rewriter) {
  Location loc = linalgOp.getLoc();
  auto ctx = linalgOp.getContext();

  auto output = linalgOp.getDpsInits()[0];
  auto outputType = cast<ShapedType>(output.getType());
  auto outputShape = outputType.getShape();
  auto outputByteWidth = outputType.getElementTypeBitWidth() / 8;

  // Create descriptors and load values for all inputs.
  SmallVector<SmallVector<Value>> loadedInputs;
  for (auto input : linalgOp.getDpsInputs()) {
    SmallVector<Value> inputTiles = createCoarseDscTiles(
        rewriter, loc, input, outputShape, /*isVnni=*/false);
    auto loadSh = determine2DTileSize(outputShape, /*isVnni=*/false, outputByteWidth);
    SmallVector<Value> loadedVals =
        loadDescTiles(rewriter, loc, inputTiles, /*hint=*/nullptr, /*vnniConf=*/std::nullopt,
                     /*transpose=*/nullptr, /*transpose_bit=*/nullptr, loadSh);
    for (auto v : loadedVals) {
      v.dump();
    }
    llvm::dbgs() << "x done\n";
    loadedInputs.push_back(loadedVals);
  }

  // Extract SIMD sized sub-tiles from loaded tiles.
  // TODO: Fetch SIMD sizes from target descriptor.
  int64_t maxSizeSIMD = 256;
  auto loadShape = cast<VectorType>(loadedInputs[0][0].getType()).getShape();
  // For sake of n-D loads and store, the vectorized operations are kept in 2D
  // shape. The loaded tiles might be larger than what SIMD units can handle.
  // Thus, split the registers into contiguous smaller slices. The current
  // hardware load restrictions ensure that the loaded tile width will not
  // exceed SIMD size.
  //
  // Take at least one whole row plus as many extra rows as can fit into
  // a single SIMD instruction.
  int64_t subTileCols = loadShape[1];
  int64_t subTileRows = std::min(loadShape[0], maxSizeSIMD / subTileCols);

  SmallVector<SmallVector<Value>> vecSubTiles;
  // NOLINTBEGIN
  for (auto inputTiles : loadedInputs) {
    loadShape = cast<VectorType>(inputTiles[0].getType()).getShape();
    llvm::dbgs() << "loadShape: " << loadShape[0] << " " << loadShape[1] << "\n";
    TilesArray subTiles =
        extractVecSubTiles(rewriter, loc, inputTiles, outputShape, loadShape,
                           {subTileRows, subTileCols});
    llvm::dbgs() << "Input extra sub-tiles:\n";
    for (auto v : inputTiles) { v.dump(); }
    llvm::dbgs() << "Output extra sub-tiles:\n";
    for (auto v : subTiles.toFlatVector()) { v.dump(); }
    vecSubTiles.push_back(subTiles.toFlatVector());
  }
  // NOLINTEND

  // Perform vectorized computations for each output tile.
  SmallVector<Value> results;
  for (size_t i = 0; i < vecSubTiles[0].size(); i++) {
    // Operands are sub-tiles at the same location.
    SmallVector<Value> operands;
    for (auto inputs : vecSubTiles) {
      operands.push_back(inputs[i]);
    }

    // Create SIMD operations on the sub-tiles.
    auto res = lowerEltwiseOp(linalgOp, operands, rewriter);
    if (!res)
      return failure();

    results.push_back(*res);
  }

  SmallVector<Value> outputTiles;
  if (hasSharedMemSpace(output))
    outputTiles = createSLMDescTiles(rewriter, loc, output, outputShape, {subTileRows, subTileCols});
    // outputTiles = createCoarseDscTiles(rewriter, loc, output, outputShape, /*isVnni=*/false);
  else
    outputTiles = createNdDescriptorTiles(
       rewriter, loc, output, outputShape, {0, 0}, {subTileRows, subTileCols});

  // Store results.
  auto writeCacheHint =
      xegpu::CachePolicyAttr::get(ctx, xegpu::CachePolicy::WRITE_BACK);

  storeDescTiles(rewriter, loc, results, outputTiles, writeCacheHint);

  // for (size_t i = 0; i < outputTiles.size(); i++) {
  //   rewriter.create<xegpu::StoreNdOp>(loc, results[i], outputTiles[i],
  //                                     /*l1_hint=*/writeCacheHint,
  //                                     /*l2_hint=*/writeCacheHint,
  //                                     /*l3_hint=*/writeCacheHint);
  // }

  rewriter.eraseOp(linalgOp);

  return success();
}

// Convert a GEMM-like operation to an XeGPU kernel.
template <typename LinalgOpTy>
struct ConvertGemmLikeToXeGPU : public OpRewritePattern<LinalgOpTy> {
  using OpRewritePattern<LinalgOpTy>::OpRewritePattern;
  // Constrain conversion to the supported GEMM-like ops.
  static_assert(
      llvm::is_one_of<LinalgOpTy, linalg::MatmulOp, linalg::BatchReduceMatmulOp,
                      linalg::GenericOp, linalg::MatmulTransposeBOp>::value);

  ConvertGemmLikeToXeGPU(MLIRContext *ctx, LinalgToXeGPUOptions options)
      : OpRewritePattern<LinalgOpTy>(ctx), options(options) {}

  LogicalResult matchAndRewrite(LinalgOpTy gemmLikeOp,
                                PatternRewriter &rewriter) const override {
    if (!gemmLikeOp.hasPureBufferSemantics()) {
      return rewriter.notifyMatchFailure(
          gemmLikeOp, "Linalg GEMM-like to GPU expects memref type");
    }
    if (gemmLikeOp.hasDynamicShape()) {
      return rewriter.notifyMatchFailure(
          gemmLikeOp, "Expect static shape when mapping to GPU");
    }

    using namespace structured_match;
    auto matmulMatcher =
        StructuredOpMatcher::make<linalg::GenericOp>()
            .operation(NumDpsInits(EqualsTo(1)))
            .operation(NumDpsInputs(EqualsTo(2)))
            .operation(NumRegions(EqualsTo(1)))
            .operation(NumOfLoops(EqualsTo(3)))
            .input(MatchAll(), HasStaticShape())
            .output(MatchAll(), HasStaticShape())
            .region(MatchOne(0), WithOpChain<arith::MulFOp, arith::AddFOp>());
    if (isa<linalg::GenericOp>(gemmLikeOp) &&
        !matmulMatcher.match(gemmLikeOp)) {
      return rewriter.notifyMatchFailure(
          gemmLikeOp, "Generic does not represent a GEMM-like operation");
    }

    for (auto input : gemmLikeOp.getDpsInputs()) {
      // 3D inputs are also acceptable in case of brgemm.
      auto isInputValid =
          isValidMemrefOperand(gemmLikeOp, input, rewriter, /*maxDims=*/3);
      if (failed(isInputValid))
        return isInputValid;
    }
    auto isOutputValid =
        isValidMemrefOperand(gemmLikeOp, gemmLikeOp.getDpsInits()[0], rewriter);
    if (failed(isOutputValid))
      return isOutputValid;

    // Ensure that reduction dimension tiling also works for smaller
    // workloads.
    auto aType = cast<ShapedType>(gemmLikeOp.getDpsInputs()[0].getType());
    auto kDim = aType.getShape().back();
    auto kTile = kDim < options.kTile ? kDim : options.kTile;

    // DPAS hardware sizes in MxNxK format.
    // TODO: In case more hardware configurations are available,
    //       add some automatic selection for optimal sizes.
    if (options.dpasTile.empty()) {
      return rewriter.notifyMatchFailure(gemmLikeOp, "Expect DPAS block sizes");
    }

    if (!isDPASCompatible(gemmLikeOp, kTile, options.dpasTile)) {
      return rewriter.notifyMatchFailure(
          gemmLikeOp, "GEMM-like compute does not fit in DPAS tiles");
    }

    return createDPASKernel(gemmLikeOp, options.dpasTile, kTile, options.stages,
                            rewriter);
  }

private:
  LinalgToXeGPUOptions options;
};

// Convert a named elementwise operation to an XeGPU kernel.
template <typename LinalgOpTy>
struct ConvertNamedEltwiseToXeGPU : public OpRewritePattern<LinalgOpTy> {
  using OpRewritePattern<LinalgOpTy>::OpRewritePattern;

  ConvertNamedEltwiseToXeGPU(MLIRContext *ctx, LinalgToXeGPUOptions options)
      : OpRewritePattern<LinalgOpTy>(ctx), options(options) {}

  LogicalResult matchAndRewrite(LinalgOpTy eltwiseOp,
                                PatternRewriter &rewriter) const override {
    if (!eltwiseOp.hasPureBufferSemantics()) {
      return rewriter.notifyMatchFailure(
          eltwiseOp, "Linalg eltwise to GPU expects memref type");
    }
    if (eltwiseOp.hasDynamicShape()) {
      return rewriter.notifyMatchFailure(
          eltwiseOp, "Expect static shape when mapping to GPU");
    }

    for (auto input : eltwiseOp.getDpsInputs()) {
      auto isInputValid = isValidMemrefOperand(eltwiseOp, input, rewriter);
      if (failed(isInputValid))
        return isInputValid;
    }
    auto isOutputValid =
        isValidMemrefOperand(eltwiseOp, eltwiseOp.getDpsInits()[0], rewriter);
    if (failed(isOutputValid))
      return isOutputValid;

    return createEltwiseKernel(eltwiseOp, rewriter);
  }

private:
  LinalgToXeGPUOptions options;
};

// Create XeGPU kernel out of memory fill operation.
LogicalResult createMemoryFillKernel(linalg::LinalgOp linalgOp,
                                     PatternRewriter &rewriter) {
  Location loc = linalgOp.getLoc();
  auto ctx = linalgOp.getContext();

  auto scalar = linalgOp.getDpsInputs()[0];
  auto output = linalgOp.getDpsInits()[0];
  auto outputType = cast<ShapedType>(output.getType());
  auto outputShape = outputType.getShape();

  if (outputShape.size() != 2) {
    return rewriter.notifyMatchFailure(
        linalgOp, "Memory fill operation expects 2D output");
  }

  // Otherwise 'xegpu-to-vc' pass will fail to convert it to VC
  if (outputShape[0] * outputShape[1] < 16) {
    return rewriter.notifyMatchFailure(
        linalgOp, "Memory fill operation is to small to be converted to xegpu");
  }

  // Extract SIMD sized sub-tiles
  int64_t maxSizeSIMD = hasSharedMemSpace(output) ? 32 : 256;
  int64_t subTileCols = std::min(outputShape[1], maxSizeSIMD);
  int64_t subTileRows =
      std::min(outputShape[0], std::max(maxSizeSIMD / subTileCols, 1L));

  SmallVector<Value> outputTiles;
  if (hasSharedMemSpace(output))
    outputTiles = createCoarseDscTiles(rewriter, loc, output, outputShape, /*isVnni=*/false);
  else
    outputTiles = createDescriptorTiles(
      rewriter, loc, output, outputShape, {0, 0}, {subTileRows, subTileCols});

  SmallVector<Value> results;
  for (size_t i = 0; i < outputTiles.size(); i++) {
    // Operands are sub-tiles at the same location.
    auto flatType = VectorType::get({subTileRows * subTileCols},
                                    outputType.getElementType());
    auto tileType = VectorType::get({subTileRows, subTileCols},
                                    outputType.getElementType());
    Value vec = rewriter.create<vector::BroadcastOp>(loc, flatType, scalar);
    Value res = rewriter.create<vector::ShapeCastOp>(loc, tileType, vec);

    if (!res)
      return failure();

    results.push_back(res);
  }

  // Store results.
  auto writeCacheHint =
      xegpu::CachePolicyAttr::get(ctx, xegpu::CachePolicy::WRITE_BACK);
  
  storeDescTiles(rewriter, loc, results, outputTiles, writeCacheHint);

  // for (size_t i = 0; i < outputTiles.size(); i++) {
  //   rewriter.create<xegpu::StoreNdOp>(loc, results[i], outputTiles[i],
  //                                     /*l1_hint=*/writeCacheHint,
  //                                     /*l2_hint=*/writeCacheHint,
  //                                     /*l3_hint=*/writeCacheHint);
  // }

  rewriter.eraseOp(linalgOp);

  return success();
}

// Convert a named fill operation to an XeGPU kernel.
template <typename LinalgOpTy>
struct ConvertMemoryFillToXeGPU : public OpRewritePattern<LinalgOpTy> {
  using OpRewritePattern<LinalgOpTy>::OpRewritePattern;

  ConvertMemoryFillToXeGPU(MLIRContext *ctx, LinalgToXeGPUOptions options)
      : OpRewritePattern<LinalgOpTy>(ctx), options(options) {}

  LogicalResult matchAndRewrite(LinalgOpTy linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!linalgOp.hasPureBufferSemantics()) {
      return rewriter.notifyMatchFailure(
          linalgOp, "Linalg eltwise to GPU expects memref type");
    }
    if (linalgOp.hasDynamicShape()) {
      return rewriter.notifyMatchFailure(
          linalgOp, "Expect static shape when mapping to GPU");
    }
    auto isInputValid =
        success(linalgOp.isScalar(linalgOp.getDpsInputOperand(0)));
    if (failed(isInputValid))
      return isInputValid;

    auto isOutputValid =
        isValidMemrefOperand(linalgOp, linalgOp.getDpsInits()[0], rewriter);
    if (failed(isOutputValid))
      return isOutputValid;

    return createMemoryFillKernel(linalgOp, rewriter);
  }

private:
  LinalgToXeGPUOptions options;
};

// TODO: Finalize BRGEMM support and register the pattern.
void populateLinalgGemmToXeGPUPatterns(RewritePatternSet &patterns,
                                       LinalgToXeGPUOptions options) {
  patterns.add<ConvertGemmLikeToXeGPU<linalg::MatmulOp>,
               ConvertGemmLikeToXeGPU<linalg::GenericOp>,
               ConvertGemmLikeToXeGPU<linalg::MatmulTransposeBOp>>(
      patterns.getContext(), options);
}

void populateLinalgEltwiseToXeGPUPatterns(RewritePatternSet &patterns,
                                          LinalgToXeGPUOptions options) {
  patterns.add<ConvertNamedEltwiseToXeGPU<linalg::AbsOp>,
               ConvertNamedEltwiseToXeGPU<linalg::AddOp>,
               ConvertNamedEltwiseToXeGPU<linalg::CeilOp>,
               ConvertNamedEltwiseToXeGPU<linalg::DivOp>,
               ConvertNamedEltwiseToXeGPU<linalg::DivUnsignedOp>,
               ConvertNamedEltwiseToXeGPU<linalg::ExpOp>,
               ConvertNamedEltwiseToXeGPU<linalg::FloorOp>,
               ConvertNamedEltwiseToXeGPU<linalg::MaxOp>,
               ConvertNamedEltwiseToXeGPU<linalg::MulOp>,
               ConvertNamedEltwiseToXeGPU<linalg::NegFOp>,
               ConvertNamedEltwiseToXeGPU<linalg::SubOp>>(patterns.getContext(),
                                                          options);
}

void populateLinalgMemoryFillToXeGPUPatterns(RewritePatternSet &patterns,
                                             LinalgToXeGPUOptions options) {
  patterns.add<ConvertMemoryFillToXeGPU<linalg::FillOp>>(patterns.getContext(),
                                                         options);
}

struct LinalgToXeGPU : public gc::impl::LinalgToXeGPUBase<LinalgToXeGPU> {
  using LinalgToXeGPUBase::LinalgToXeGPUBase;

  void runOnOperation() override {
    LinalgToXeGPUOptions options{
        kTile, stages, SmallVector<int64_t>(dpasTile.begin(), dpasTile.end())};

    // Run GEMM pattern first to allow fusion with its consumers.
    RewritePatternSet gemmPatterns(&getContext());
    populateLinalgGemmToXeGPUPatterns(gemmPatterns, options);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(gemmPatterns));

    // Convert memory fill ops.
    RewritePatternSet fillPatterns(&getContext());
    populateLinalgMemoryFillToXeGPUPatterns(fillPatterns, options);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(fillPatterns));

    // Convert other remaining ops.
    RewritePatternSet patterns(&getContext());
    populateLinalgEltwiseToXeGPUPatterns(patterns, options);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace
