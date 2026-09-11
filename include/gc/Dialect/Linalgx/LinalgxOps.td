//===- LinalgxOps.td - linalgx dialect ops -----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LINALGX_OPS
#define LINALGX_OPS

include "LinalgxDialect.td"

include "mlir/Dialect/Linalg/IR/LinalgInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"

// Base class for Linalg dialect ops that do not correspond to library calls.
class Linalgx_Op<string mnemonic, list<Trait> traits = []> :
    Op<LinalgxDialect, mnemonic, traits>;

def Linalgx_AttentionOp : Linalgx_Op<"attention",
    [
     DeclareOpInterfaceMethods<IndexingMapOpInterface, ["getMatchingIndexingMap"]>,
     DeclareOpInterfaceMethods<DestinationStyleOpInterface>,
     DeclareOpInterfaceMethods<AggregatedOpInterface, ["decomposeOperation"]>,
     DeclareOpInterfaceMethods<TilingInterface,
      ["getIterationDomain",
       "getLoopIteratorTypes",
       "getResultTilePosition",
       "getTiledImplementation",
       "generateResultTileValue"]>]> {
  let summary = [{Attention operator.}];
  let description = [{
    Computes the scaled dot product attention function:

    attention(Q, K, V, scale) = softmax(Q @ K.T * scale) @ V

    Here Q, K, V are given tensors and scale is a scalar value specifying
    the scale to use.

    If an additional mask argument M is included, the result of the first matmul is modified according to:

    Q @ K.T += M
  }];

  let arguments = (ins AnyShaped:$query,
                       AnyShaped:$key,
                       AnyShaped:$value,
                       AnyFloat:$scale,
                       Optional<AnyShaped>:$mask,
                       AnyShaped:$output,
                       AffineMapArrayAttr:$indexing_maps,
                       OptionalAttr<DictionaryAttr>:$decomposition_config
  );

  let results = (outs Variadic<AnyRankedTensor>:$results);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = [{
    attr-dict
    `ins` `(` $query `,` $key `,` $value `,` $scale (`,` $mask^)?  `:` type($query) `,` type($key) `,` type($value) `,` type($scale) (`,` type($mask)^ )?`)`
    `outs` `(` $output `:` type($output) `)`
    (`->` type($results)^)?
  }];

  let builders = [
    OpBuilder<(ins "TypeRange":$results,
      "Value":$query,
      "Value":$key,
      "Value":$value,
      "Value":$scale,
      "Value":$output,
      "ArrayAttr":$indexing_maps,
      CArg<"std::optional<Value>", "std::nullopt">:$mask)>,
    OpBuilder<(ins "TypeRange":$results,
      "ValueRange":$inputOperands,
      "ValueRange":$initOperands,
      "ArrayAttr":$indexing_maps)>
  ];

  let extraClassDeclaration = [{
    SmallVector<AffineMap> getIndexingMapsArray();

    ArrayRef<int64_t> getShape(OpOperand* opOperand);

    AffineMap getQueryMap() {
      return cast<AffineMap>(getIndexingMapsArray()[0]);
    }
    AffineMap getKeyMap() {
      return cast<AffineMap>(getIndexingMapsArray()[1]);
    }
    AffineMap getValueMap() {
      return cast<AffineMap>(getIndexingMapsArray()[2]);
    }
    AffineMap getScaleMap() {
      return cast<AffineMap>(getIndexingMapsArray()[3]);
    }
    std::optional<AffineMap> getMaskMap() {
      if (getMask()) {
        return cast<AffineMap>(getIndexingMapsArray()[4]);
      }
      return std::nullopt;
    }
    AffineMap getOutputMap() {
      return cast<AffineMap>(getIndexingMapsArray()[getNumDpsInputs()]);
    }
    int64_t getIterationDomainRank() {
      return getQueryMap().getNumDims();
    }

    /* Decomposition control attributes */

    // Attributes to set on QK and PV matmul after decomposition.
    static StringRef getQKAttrStr() { return "qk_attrs"; }
    static StringRef getPVAttrStr() { return "pv_attrs"; }
  }];

  let hasCanonicalizer = 1;
}

def Linalgx_GroupedMatmulOp : Linalgx_Op<"grouped_matmul",
    [
     DeclareOpInterfaceMethods<IndexingMapOpInterface, ["getMatchingIndexingMap"]>,
     DeclareOpInterfaceMethods<DestinationStyleOpInterface>,
     DeclareOpInterfaceMethods<TilingInterface,
      ["getIterationDomain",
       "getLoopIteratorTypes",
       "getResultTilePosition",
       "getTiledImplementation",
       "generateResultTileValue"]>]> {
  let summary = [{Grouped (per-expert) matmul with a runtime weight selector.}];
  let description = [{
    Computes, for every slot `s` and every token `m`:

      out[s, m, :] = lhs[m, :] * weights[indices[m, s], :, :]^T

    i.e. a matmul whose weight matrix is picked at runtime out of `weights`'
    leading (expert) dimension. In a MoE top-k routing the same token row feeds
    every slot, so `lhs` does not have a slot dimension at all.

    Operand ranks: `lhs` [M, K], `weights` [E, N, K] (transposed, i.e. the
    reduction dim is the innermost one), `indices` [M, S] (i32/i64),
    `output` [S, M, N].

    Iteration domain is `(slot, m, n, k)` where `k` is the only reduction dim.
    Note that `weights` is indexed by `(n, k)` only: its leading dimension is
    addressed through `indices` and is therefore not part of the domain.

    Keeping `lhs` 2-D is what makes the tiled form a plain `linalg.matmul`: the
    slot dim is tiled to 1, so `weights` and `output` are sliced rank-reducingly
    and every operand of the tile is 2-D. See `getTiledImplementation()`.

    CONTRACT, to be guaranteed by whoever produces `indices`: all tokens of one
    `tile_m`-sized block must share the same expert, i.e. `indices[m, s]` is
    constant for `m` in `[b * tile_m, (b + 1) * tile_m)`. The tiled
    implementation reads the expert index once per tile, so the op is only
    correct for inputs whose rows are already grouped by expert. `tile_m` is
    therefore a contract, not a free tiling parameter: the tiling pass must use
    exactly this size along `m`.
  }];

  let arguments = (ins AnyShaped:$lhs,
                       AnyShaped:$weights,
                       AnyShaped:$indices,
                       AnyShaped:$output,
                       I64Attr:$tile_m,
                       AffineMapArrayAttr:$indexing_maps
  );

  let results = (outs Variadic<AnyRankedTensor>:$results);
  let hasVerifier = 1;
  let assemblyFormat = [{
    attr-dict
    `ins` `(` $lhs `,` $weights `,` $indices `:` type($lhs) `,` type($weights) `,` type($indices) `)`
    `outs` `(` $output `:` type($output) `)`
    (`->` type($results)^)?
  }];

  let extraClassDeclaration = [{
    SmallVector<AffineMap> getIndexingMapsArray();

    ArrayRef<int64_t> getShape(OpOperand* opOperand);

    AffineMap getLhsMap() { return getIndexingMapsArray()[0]; }
    AffineMap getWeightsMap() { return getIndexingMapsArray()[1]; }
    AffineMap getIndicesMap() { return getIndexingMapsArray()[2]; }
    AffineMap getOutputMap() { return getIndexingMapsArray()[3]; }

    int64_t getIterationDomainRank() { return getLhsMap().getNumDims(); }

    /* Iteration domain dim positions. */
    static constexpr int64_t kSlotDim = 0;
    static constexpr int64_t kMDim = 1;
    static constexpr int64_t kNDim = 2;
    static constexpr int64_t kKDim = 3;
  }];
}

#endif // LINALGX_OPS
