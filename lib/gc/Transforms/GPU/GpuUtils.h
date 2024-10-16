//===-- GpuUtils.h - DESC ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef GPUUTILS_H
#define GPUUTILS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

using namespace mlir;

template <typename DerivedT> struct GpuPass {

  size_t getDeviceProperty(Builder &builder, StringRef name,
                           size_t defaultValue) {
    if (auto mod = static_cast<DerivedT *>(this)
                       ->getOperation()
                       ->template getParentOfType<ModuleOp>()) {
      DataLayout layout(mod);
      if (auto value = layout.getDevicePropertyValue(
              builder.getStringAttr("GPU" /* device ID*/),
              builder.getStringAttr(name))) {
        if (auto attr = dyn_cast<IntegerAttr>(*value)) {
          return static_cast<size_t>(attr.getInt());
        }
      }
    }
    return defaultValue;
  }

  size_t getEuMem(Builder &builder) {
    return getDeviceProperty(builder, "L1_cache_size_in_bytes",
                             static_cast<DerivedT *>(this)->euMem);
  }

  size_t getEuThreads(Builder &builder) {
    return getDeviceProperty(builder, "threads_per_eu",
                             static_cast<DerivedT *>(this)->euThreads);
  }
};

template <typename A, typename B>
static IntegerAttr ceil(OpBuilder &builder, A a, B b) {
  return builder.getIndexAttr(
      static_cast<int64_t>(std::ceil(static_cast<double>(a) / b)));
}

static int64_t getConstIdxValue(Value value) {
  if (auto op = value.getDefiningOp<arith::ConstantIndexOp>()) {
    return op.value();
  }
  if (auto minOp = value.getDefiningOp<affine::AffineMinOp>()) {
    for (const AffineExpr &result : minOp.getMap().getResults()) {
      if (auto constExpr = dyn_cast<AffineConstantExpr>(result)) {
        return constExpr.getValue();
      }
    }
  }
  if (auto minOp = value.getDefiningOp<arith::MinSIOp>()) {
    for (Value operand : {minOp.getLhs(), minOp.getRhs()}) {
      if (auto v = getConstIdxValue(operand))
        return v;
    }
  }
  return 0;
}
#endif
