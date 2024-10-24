//===-- GpuTilingAndFusion.cpp - DESC ---------------------------*- C++ -*-===//
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
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/RegionUtils.h"

#include "./GpuUtils.h"
#include "gc/Utils/Log.h"

using namespace mlir;
using namespace mlir::gc;

namespace mlir::gc {
#define GEN_PASS_DECL_GPUTILINGANDFUSION
#define GEN_PASS_DEF_GPUTILINGANDFUSION
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

struct GpuTilingAndFusion final
    : GpuPass<GpuTilingAndFusion>,
      gc::impl::GpuTilingAndFusionBase<GpuTilingAndFusion> {
  friend struct GpuPass;
  explicit GpuTilingAndFusion()
      : GpuTilingAndFusion(GpuTilingAndFusionOptions{}) {}
  explicit GpuTilingAndFusion(const GpuTilingAndFusionOptions &opts)
      : GpuPass(), GpuTilingAndFusionBase(opts) {}

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    scf::SCFTileAndFuseOptions opts;
    auto numEus = getNumEus(rewriter);
    auto numEusPerSlice = getNumEusPerSlice(rewriter);
    auto numThreadsPerEu = getNumThreadsPerEu(rewriter);
    auto cacheSize = getCacheSize(rewriter);
    auto vectorWidth = getVectorWidth(rewriter);
    auto cachePerThread =
        std::max(cacheSize / numEusPerSlice / numThreadsPerEu, vectorWidth);
    opts.tilingOptions.setTileSizeComputationFunction(
        [cachePerThread, vectorWidth, numThreads = numEus * numThreadsPerEu](
            OpBuilder &builder, Operation *op) -> SmallVector<OpFoldResult> {
          auto ti = dyn_cast<TilingInterface>(op);
          if (!ti) {
            return {};
          }

          auto itTypes = ti.getLoopIteratorTypes();
          auto itDomains = ti.getIterationDomain(builder);
          assert(itTypes.size() == itDomains.size());

          SmallVector<int64_t> tiles;
          int64_t numIterations = 1;
          for (auto [t, r] : zip(itTypes, itDomains)) {
            if (t == utils::IteratorType::parallel) {
              if (auto v = getConstantIntValue(r.size)) {
                numIterations *= *v;
                tiles.emplace_back(*v);
              } else {
                return computeDynamicTiles(builder, ti, numThreads,
                                           cachePerThread);
              }
            }
          }

          if (tiles.empty()) {
            return {};
          }

          auto elementSize = getElementSize(op);
          auto sizePerThread = numIterations / numThreads * elementSize;
          auto totalSize = std::max(sizePerThread, cachePerThread);
          totalSize = std::max(totalSize / elementSize, 64L);

          // If the operation could be lowered to XeGPU, make the tiles
          // proportional to the vector width.
          if (canLowerToXeGPU(op)) {
            totalSize = std::max(totalSize / vectorWidth, 1L) * vectorWidth;
          }

          adjustTiles(totalSize, tiles);

          unsigned counter = 0;
          SmallVector<OpFoldResult> result;
          result.reserve(itDomains.size());

          for (auto [t, r] : zip(itTypes, itDomains)) {
            if (t != utils::IteratorType::parallel) {
              result.emplace_back(builder.getIndexAttr(0));
            } else {
              result.emplace_back(builder.getIndexAttr(tiles[counter++]));
            }
          }

          return result;
        });
    opts.setFusionControlFn(
        [&](tensor::ExtractSliceOp, OpResult originalProducer, bool)
            -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
          Operation *op = originalProducer.getOwner();
          if (!op) {
            return std::nullopt;
          }
          if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
            if (!linalgOp.hasOnlyProjectedPermutations()) {
              return std::nullopt;
            }
          }
          return scf::SCFTileAndFuseOptions::ControlFnResult{};
        });
    opts.tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);

    auto fn = getOperation();
    tileAndFuse(fn, rewriter, opts);
  }

private:
  static void tileAndFuse(Operation *op, RewriterBase &rewriter,
                          const scf::SCFTileAndFuseOptions &opts) {
    for (auto ti = findTi(op); ti; ti = findTi(op)) {
      auto result =
          scf::tileConsumerAndFuseProducersUsingSCF(rewriter, *ti, opts);

      if (failed(result)) {
        ti->emitError() << "Failed to tile and fuse using SCF";
        return;
      }

      SmallVector<Operation *> opsToReplace{ti->getOperation()};
      append_range(opsToReplace, result->fusedProducers);
      for (Operation *toReplace : opsToReplace) {
        if (toReplace->getParentOp() == nullptr) {
          continue;
        }

        for (OpResult res : toReplace->getResults()) {
          if (auto repl = result->replacements.lookup(res)) {
            rewriter.replaceAllUsesWith(res, repl);
          }
        }

        if (failed(simplifyRegions(rewriter, op->getRegions()))) {
          gcLogD("Failed to simplify regions");
        }

        if (toReplace->getParentOp() == nullptr) {
          continue;
        }

        // For some reason (probably a bug?) the operation could be
        // referenced by a dead code inside the replacement, that prevents
        // this operation from being erased. Erasing the dead code first.
        for (auto u : toReplace->getUsers()) {
          if (u->use_empty()) {
            rewriter.eraseOp(u);
          }
        }

        if (toReplace->use_empty()) {
          rewriter.eraseOp(toReplace);
        } else {
          gcLogE("Unable to erase operation!");
        }
      }
    }
  }

  static std::optional<TilingInterface> findTi(Operation *op) {
    std::optional<TilingInterface> last;
    op->walk<WalkOrder::PreOrder>([&](linalg::LinalgOp linalgOp) {
      if (linalgOp.hasOnlyProjectedPermutations() &&
          !linalgOp->getParentOfType<scf::ForallOp>()) {
        if (auto ti = dyn_cast<TilingInterface>(linalgOp.getOperation())) {
          last = ti;
        }
      }
      return WalkResult::skip();
    });
    return last;
  }

  static SmallVector<OpFoldResult> computeDynamicTiles(OpBuilder &builder,
                                                       TilingInterface ti,
                                                       int64_t numThreads,
                                                       int64_t cachePerThread) {
    auto itTypes = ti.getLoopIteratorTypes();
    auto itDomains = ti.getIterationDomain(builder);
    assert(itTypes.size() == itDomains.size());

    auto loc = ti.getLoc();
    Value dynamicSize;
    auto staticSize = getElementSize(ti.getOperation());
    unsigned loopCount = 0;

    for (auto [t, r] : zip(itTypes, itDomains)) {
      if (t != utils::IteratorType::parallel) {
        continue;
      }
      loopCount++;
      if (auto v = getConstantIntValue(r.size)) {
        staticSize *= *v;
      } else if (dynamicSize) {
        dynamicSize = builder.create<arith::MulIOp>(loc, dynamicSize,
                                                    r.size.get<Value>());
      } else {
        dynamicSize = r.size.get<Value>();
      }
    }

    assert(loopCount);
    assert(dynamicSize);
    if (staticSize > 1) {
      dynamicSize = builder.create<arith::MulIOp>(
          loc, dynamicSize,
          builder.create<arith::ConstantIndexOp>(loc, staticSize));
    }
    auto i64Type = builder.getI64Type();
    dynamicSize = builder.create<arith::UIToFPOp>(
        loc, builder.getF64Type(),
        builder.create<arith::IndexCastOp>(loc, i64Type, dynamicSize));

    // TODO: Call the adjustTiles() function for the tiles calculation.

    auto nt = builder.create<arith::ConstantFloatOp>(
        loc, APFloat(static_cast<double>(numThreads)), builder.getF64Type());
    auto cpt = builder.create<arith::ConstantFloatOp>(
        loc, APFloat(static_cast<double>(cachePerThread)),
        builder.getF64Type());
    Value totalSize = builder.create<arith::MaximumFOp>(
        loc, builder.getF64Type(),
        builder.create<arith::DivFOp>(loc, dynamicSize, nt), cpt);
    auto pow = builder.create<arith::ConstantFloatOp>(
        loc, APFloat(1.0 / loopCount), builder.getF64Type());
    // The average tile size is totalSize^(1 / loopCount)
    Value avgTileSize = builder.create<math::PowFOp>(loc, totalSize, pow);
    avgTileSize = builder.create<arith::MaximumFOp>(
        loc, builder.getF64Type(),
        builder.create<arith::ConstantFloatOp>(loc, APFloat(1.0),
                                               builder.getF64Type()),
        avgTileSize);
    avgTileSize = builder.create<arith::FPToSIOp>(loc, i64Type, avgTileSize);

    SmallVector<OpFoldResult> tiles;
    tiles.reserve(itDomains.size());

    for (auto [t, r] : zip(itTypes, itDomains)) {
      if (t != utils::IteratorType::parallel) {
        tiles.emplace_back(builder.getIndexAttr(1));
      } else {
        Value value;
        if (auto v = getConstantIntValue(r.size)) {
          value = builder.create<arith::ConstantIntOp>(loc, *v, i64Type);
        } else {
          value = builder.create<arith::IndexCastOp>(loc, i64Type,
                                                     r.size.get<Value>());
        }
        value =
            builder.create<arith::MinSIOp>(loc, i64Type, value, avgTileSize);
        tiles.emplace_back(builder.create<arith::IndexCastOp>(
            loc, builder.getIndexType(), value));
      }
    }

    return tiles;
  }

  static int64_t getElementSize(Operation *op) {
    int64_t elementSize = 1;
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      if (auto inits = linalgOp.getDpsInits(); !inits.empty()) {
        if (auto t = getElementTypeOrSelf(inits[0].getType());
            t.isIntOrFloat()) {
          elementSize = t.getIntOrFloatBitWidth() / 8;
        }
      }
    }
    return elementSize;
  }

  // TODO: Add more checks
  static bool canLowerToXeGPU(Operation *operation) {
    auto op = dyn_cast<linalg::LinalgOp>(operation);
    if (!op) {
      return false;
    }
    if (op.hasDynamicShape()) {
      return false;
    }

    auto checkOperand = [&](Value operand, bool isOutput = false) {
      ShapedType type;
      if (auto memref = dyn_cast<MemRefType>(operand.getType())) {
        type = memref;
      } else if (auto tensor = dyn_cast<RankedTensorType>(operand.getType())) {
        type = tensor;
      } else {
        return false;
      }

      auto shape = type.getShape();
      if (isOutput) {
        if (shape.size() != 2 || shape[0] * shape[1] < 16) {
          return false;
        }
      } else if (shape.size() > 2) {
        return false;
      }

      return true;
    };

    if (auto inits = op.getDpsInits();
        !inits.empty() && !checkOperand(inits[0], true)) {
      return false;
    }

    if (auto inputs = op.getDpsInputs();
        !std::all_of(inputs.begin(), inputs.end(),
                     [&](Value v) { return checkOperand(v); })) {
      return false;
    }

    return true;
  }
};
} // namespace
