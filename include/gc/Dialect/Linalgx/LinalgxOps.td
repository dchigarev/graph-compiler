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
  let summary = [{Ragged (per-expert) matmul over group-partitioned rows.}];
  let description = [{
    A jagged / segmented batched matmul: the batch dimension is the expert, but
    the number of rows per expert is a runtime value carried by `offsets` rather
    than part of any shape. This mirrors oneDNN's `grouped_gemm` and
    `torch.nn.functional.grouped_mm`.

    For every expert `e` and every row `r` of its group:

      start(e) = e == 0 ? 0 : offsets[e - 1]
      end(e)   = offsets[e]
      out[start(e) + r, :] = lhs[start(e) + r, :] * weights[e, :, :]^T
                             for r in [0, end(e) - start(e))

    Operands:
      `lhs`             [total, K]     rows of all groups, concatenated
      `weights`         [E, N, K]      dense, transposed (K innermost)
      `offsets`         [E] i32/i64    cumulative *ends* of the groups
      `max_group_size`  index          upper bound on `end(e) - start(e)`
      `output`          [total, N]     same row partitioning as `lhs`

    `offsets` holds exclusive prefix ends with an implicit leading zero, exactly
    like oneDNN: counts [1, 3, 5] are passed as [1, 4, 9], `offsets[E-1]` equals
    `total`, and an inactive expert is a zero-length group (`offsets[e] ==
    offsets[e-1]`). Rows are NOT padded to any granularity - the tiled
    implementation clamps the row tile to the group's real length, so a tile
    never crosses a group boundary and the M tile size is a free tiling
    parameter.

    In a MoE layer the caller flattens top-k routing into `total = tokens * k`
    rows, sorted by expert, and un-permutes plus applies the router weights
    afterwards. The op has no notion of top-k slots.

    Iteration domain is `(expert, m, n, k)` with `k` the only reduction dim,
    where `m` is *group-relative*: the tiled implementation adds `start(e)` when
    slicing `lhs` and `output`. The domain is rectangular - `m` runs over
    `max_group_size` - so tiles of short groups are clamped to zero size and do
    nothing. Pass `total` for `max_group_size` if no tighter bound is known; the
    result is the same, only the loop nest is over-provisioned.

    The expert dim must be tiled to 1, which makes every operand of the tile 2-D
    and the tile itself a plain `linalg.matmul`. See `getTiledImplementation()`.
  }];

  let arguments = (ins AnyShaped:$lhs,
                       AnyShaped:$weights,
                       AnyShaped:$offsets,
                       Index:$max_group_size,
                       AnyShaped:$output,
                       AffineMapArrayAttr:$indexing_maps
  );

  let results = (outs Variadic<AnyRankedTensor>:$results);
  let hasVerifier = 1;
  let assemblyFormat = [{
    attr-dict
    `ins` `(` $lhs `,` $weights `,` $offsets `,` $max_group_size `:`
              type($lhs) `,` type($weights) `,` type($offsets) `,`
              type($max_group_size) `)`
    `outs` `(` $output `:` type($output) `)`
    (`->` type($results)^)?
  }];

  let extraClassDeclaration = [{
    SmallVector<AffineMap> getIndexingMapsArray();

    ArrayRef<int64_t> getShape(OpOperand* opOperand);

    /* `indexing_maps` holds one map per *shaped* operand, in operand order;
       `max_group_size` is a scalar and has no map. */
    AffineMap getLhsMap() { return getIndexingMapsArray()[0]; }
    AffineMap getWeightsMap() { return getIndexingMapsArray()[1]; }
    AffineMap getOffsetsMap() { return getIndexingMapsArray()[2]; }
    AffineMap getOutputMap() { return getIndexingMapsArray()[3]; }

    int64_t getIterationDomainRank() { return getLhsMap().getNumDims(); }

    /* Iteration domain dim positions. */
    static constexpr int64_t kExpertDim = 0;
    static constexpr int64_t kMDim = 1;
    static constexpr int64_t kNDim = 2;
    static constexpr int64_t kKDim = 3;

    /* Operand numbers. */
    static constexpr unsigned kMaxGroupSizeOperand = 3;
  }];
}

#endif // LINALGX_OPS
