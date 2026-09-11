#include "TilingUtils.h"
#include "gc/Dialect/Linalgx/LinalgxDialect.h"
#include "gc/Dialect/Linalgx/LinalgxOps.h"

namespace mlir::gc {
#define GEN_PASS_DECL_TILEGROUPEDMATMUL
#define GEN_PASS_DEF_TILEGROUPEDMATMUL
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

// Tiles 'linalgx.grouped_matmul' at the work-group level and the contraction it
// decomposes into at the sub-group level.
//
// The pass must run *after* TileContraction: by then every plain contraction is
// already marked with 'gc.tiling.level', so the WG search only picks up
// grouped_matmul ops, while the sub-group search still finds the contraction
// this op's getTiledImplementation() emits inside the freshly created WG loop.
struct TileGroupedMatmul final
    : TilingPass<gc::impl::TileGroupedMatmulBase<TileGroupedMatmul>> {

  bool isSupportedOp(TilingInterface ti) override {
    return isa<linalgx::GroupedMatmulOp>(ti) || isMatmulOp(ti);
  }

  // The parallel domain is (slot, m, n), so 'm' is the height dim and 'n' the
  // width dim of the generic block-size search. Pin the height to 'tile_m': it
  // is a correctness contract of the op, not a tuning knob.
  void collectTileConstraints(Target &tg, DimConstraint &width,
                              DimConstraint &height) override {
    TilingPass::collectTileConstraints(tg, width, height);
    if (auto gmm = dyn_cast<linalgx::GroupedMatmulOp>(tg.op.getOperation())) {
      auto tileM = static_cast<size_t>(gmm.getTileM());
      height.narrow(tileM);
      height.align(tileM);
    }
  }

  void computeWgTiles(Target &tg) override {
    // Run the generic search so that the M/N tiles are actually distributed
    // over sub-groups (the tile emitted by getTiledImplementation() is a plain
    // 2-D matmul, so it wants a real 2-D sg layout).
    TilingPass::computeWgTiles(tg);
    auto gmm = dyn_cast<linalgx::GroupedMatmulOp>(tg.op.getOperation());
    if (!gmm) return;

    using GMM = linalgx::GroupedMatmulOp;
    // The expert index is read once per tile, which is only uniform within a
    // single slot and a single 'tile_m' block. The constraint above normally
    // makes the search land on 'tile_m' already, but it is dropped by the
    // non-strict fallback pass, so enforce it here.
    tg.tiles[GMM::kSlotDim] = 1;
    tg.sgTiles[GMM::kSlotDim] = 1;
    auto tileM = static_cast<size_t>(gmm.getTileM());
    tg.tiles[GMM::kMDim] = tileM;
    while (tg.sgTiles[GMM::kMDim] > 1 && tileM % tg.sgTiles[GMM::kMDim])
      tg.sgTiles[GMM::kMDim] /= 2;
  }

  void computeSgTiles(Target &tg) override {
    if (isa<linalgx::GroupedMatmulOp>(tg.op.getOperation())) {
      // Nothing to do: the reduction loop is created when the contraction
      // emitted by getTiledImplementation() is tiled.
      std::fill(tg.tiles.begin(), tg.tiles.end(), 0);
      return;
    }
    TilingPass::computeSgTiles(tg);
  }
};

} // namespace
