#include "TilingUtils.h"
#include "gc/Dialect/Linalgx/LinalgxDialect.h"
#include "gc/Dialect/Linalgx/LinalgxOps.h"

namespace mlir::gc {
#define GEN_PASS_DECL_TILEGROUPEDMATMUL
#define GEN_PASS_DEF_TILEGROUPEDMATMUL
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {

// N-dim work-group tile used for 'linalgx.grouped_matmul'. Deliberately fixed
// for now; the M tile is dictated by the op ('tile_m' attribute) and K is left
// to the sub-group/reduction level.
constexpr size_t kNTile = 32;

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

  void computeWgTiles(Target &tg) override {
    auto gmm = dyn_cast<linalgx::GroupedMatmulOp>(tg.op.getOperation());
    if (!gmm) {
      TilingPass::computeWgTiles(tg);
      return;
    }
    // Iteration domain is (slot, m, n, k); 0 means "do not tile this dim".
    SmallVector<size_t> tiles(tg.tiles.size(), 0);
    SmallVector<size_t> sgTiles(tg.tiles.size(), 1);
    tiles[linalgx::GroupedMatmulOp::kSlotDim] = 1;
    tiles[linalgx::GroupedMatmulOp::kMDim] =
        static_cast<size_t>(gmm.getTileM());
    tiles[linalgx::GroupedMatmulOp::kNDim] = kNTile;
    tg.setTiles(tiles, sgTiles, /*reduction=*/false);
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
