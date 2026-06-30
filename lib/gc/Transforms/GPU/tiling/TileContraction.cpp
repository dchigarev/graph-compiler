#include "TilingUtils.h"
#include "mlir/Support/LLVM.h"
#include <algorithm>
#include <cmath>
#include <functional>

namespace mlir::gc {
#define GEN_PASS_DECL_TILECONTRACTION
#define GEN_PASS_DEF_TILECONTRACTION
#include "gc/Transforms/Passes.h.inc"
} // namespace mlir::gc

namespace {
struct TileContraction final
    : TilingPass<gc::impl::TileContractionBase<TileContraction>> {

  bool isSupportedOp(TilingInterface ti) override { return isMatmulOp(ti); }

  virtual std::tuple<SmallVector<unsigned>, SmallVector<unsigned>,
                     SmallVector<unsigned>, unsigned, unsigned>
  getSupportedBlockSizes(Target &tg, bool reduction, size_t width,
                         size_t height) override {
    if (reduction) {
      auto tiles = tg.kernelAttrs.getTiles().value();
      return std::make_tuple(
          SmallVector<unsigned>{static_cast<unsigned>(tiles.back())},
          SmallVector<unsigned>{1}, SmallVector<unsigned>{1}, 1, 1);
    }

    auto ua = tg.devAttrs.getUarch();
    auto instr =
        dyn_cast<xegpu::uArch::SubgroupMatrixMultiplyAcc>(ua->getInstruction(
            xegpu::uArch::InstructionKind::SubgroupMatrixMultiplyAcc));
    assert(instr);
    auto elType =
        cast<ShapedType>(tg.op.getOperation()->getOperand(0).getType())
            .getElementType();
    auto [widths, heights, counts, sgMul, wgMul] =
        TilingPass::getSupportedBlockSizes(tg, false, width, height);
    auto supportedM = instr->getSupportedM(elType);
    auto supportedN = instr->getSupportedN(elType);
    auto supportedK =
        llvm::map_to_vector(instr->getSupportedK(elType), [](uint32_t v) {
          return static_cast<unsigned>(v);
        });
    llvm::sort(supportedK, std::greater<unsigned>());

    size_t kSize = tg.sizes.back();
    unsigned mul = std::sqrt(getSgSize(tg));
    std::function<bool(size_t, unsigned)> mMatch;
    std::function<bool(size_t, unsigned)> nMatch;
    if (tg.sizes[tg.sizes.size() - 3] == height) {
      mMatch = [height](size_t m, unsigned mul) {
        return height % (m * mul) == 0;
      };
    } else {
      mMatch = [height](size_t m, unsigned mul) { return height == (m * mul); };
      mul = std::max<unsigned>(mul, height);
    }
    if (tg.sizes[tg.sizes.size() - 2] == width) {
      nMatch = [width](size_t n, unsigned mul) {
        return width % (n * mul) == 0;
      };
    } else {
      nMatch = [width](size_t n, unsigned mul) { return width == (n * mul); };
      mul = std::max<unsigned>(mul, width);
    }
    widths = llvm::filter_to_vector(
        widths, [&](unsigned c) { return llvm::is_contained(supportedN, c); });
    heights = llvm::filter_to_vector(
        heights, [&](unsigned c) { return llvm::is_contained(supportedM, c); });
    counts = {1};

    for (; mul >= 1; mul /= 2)
      for (auto m : heights)
        for (auto n : widths)
          for (auto k : supportedK)
            if (kSize % (k * mul) == 0 && mMatch(m, mul) && nMatch(n, mul)) {
              heights = {m};
              widths = {n};
              supportedK = {k};
              goto setTiles;
            }
  setTiles:
    SmallVector<size_t> tiles = {heights.front() * mul, widths.front() * mul,
                                 supportedK.front() * mul};
    tg.kernelAttrs.setTiles(tiles);
    return std::make_tuple(widths, heights, counts, 1, mul * mul);
  }

  SmallVector<size_t> computeThreads(Target &tg) override {
    auto sgSize = getSgSize(tg);
    auto threads = TilingPass::computeThreads(tg);
    threads[0] = std::max<size_t>(sgSize, threads[0] / sgSize);
    return threads;
  }
};
} // namespace