//===-- JitEngine.h - JIT engine wrapper ------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace mlir {
class ExecutionEngine;
} // namespace mlir
namespace llvm::orc {
class LLJIT;
} // namespace llvm::orc

namespace mlir::gc {
class JitEngine {
public:
  explicit JitEngine(std::unique_ptr<mlir::ExecutionEngine> eng);
  explicit JitEngine(std::unique_ptr<llvm::orc::LLJIT> jit);
  ~JitEngine();
  JitEngine(JitEngine &&) = default;

  bool saveToFile(llvm::StringRef filename);
  static std::unique_ptr<JitEngine>
  loadFromFile(llvm::SmallString<128> &cacheDir,
               llvm::ArrayRef<llvm::StringRef> sharedLibPaths = {});

  llvm::Expected<void *> lookup(llvm::StringRef name) const;
  llvm::Expected<void (*)(void **)> lookupPacked(llvm::StringRef name) const;

  // Fn is of type SymbolMap (&)(MangleAndInterner).
  template <typename Fn> void registerSymbols(Fn &&fn);

private:
  std::unique_ptr<mlir::ExecutionEngine> engine;
  std::unique_ptr<llvm::orc::LLJIT> lljit;
};

} // namespace mlir::gc
