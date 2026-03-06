#include "TilingUtils.h"
#include "gc/Dialect/Linalgx/LinalgxOps.h"
#include "gc/Dialect/Linalgx/LinalgxDialect.h"

namespace mlir::gc {
#define GEN_PASS_DECL_TILEATTENTION
#define GEN_PASS_DEF_TILEATTENTION
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {
struct TileAttention final
    : TilingPass<gc::impl::TileAttentionBase<TileAttention>> {

  bool isSupportedOp(TilingInterface ti) override { return isa<linalgx::AttentionOp>(ti); }

  size_t getSgSize(Target &tg) override {
    return tg.devAttrs.getUarch()->getSubgroupSize();
  }

  void computeSgTiles(Target &tg) override {
    for (size_t i = 0; i < tg.tiles.size(); ++i) {
      tg.tiles[i] = 0;
    }
  }

  void computeWgTiles(Target &tg) override {
    tg.tiles[0] = 1;
    tg.tiles[1] = 128;
    for (size_t i = 2; i < tg.tiles.size(); ++i) {
      tg.tiles[i] = 0;
    }
    // auto ua = tg.devAttrs.getUarch();
    // auto instr =
    //     dyn_cast<xegpu::uArch::SubgroupMatrixMultiplyAcc>(ua->getInstruction(
    //         xegpu::uArch::InstructionKind::SubgroupMatrixMultiplyAcc));
    // auto inputType =
    //     cast<ShapedType>(tg.op.getOperation()->getOperand(0).getType());
    // auto supportedK = instr->getSupportedK(inputType.getElementType());

    // if (tg.mode == Mode::Reduction) {
    //   tg.tiles[0] = findClosestDiv(supportedK, tg.tiles[0]);
    //   return;
    // } else if (tg.mode == Mode::Both) {
    //   tg.tiles[2] = findClosestDiv(supportedK, tg.tiles[2]);
    // }

    // auto supportedM = instr->getSupportedM(inputType.getElementType());
    // auto supportedN = instr->getSupportedN(inputType.getElementType());
    // tg.tiles[0] = findClosestDiv(supportedM, tg.tiles[0]);
    // tg.tiles[1] = findClosestDiv(supportedN, tg.tiles[1]);
  }
};
} // namespace