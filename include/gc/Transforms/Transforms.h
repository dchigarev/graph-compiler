//===- Transforms.h ----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_TRANSFORMS_TRANSFORMS_H
#define TPP_TRANSFORMS_TRANSFORMS_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {

namespace linalg {
void populateLinalgDeGeneralizationPatterns(RewritePatternSet &patterns);
} // namespace linalg
} // namespace mlir

#endif
