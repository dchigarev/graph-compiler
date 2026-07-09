//===-- JitEngine.cpp - JIT engine wrapper ----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gc/ExecutionEngine/JitEngine.h"

#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"

using namespace llvm;
using namespace llvm::orc;

namespace mlir::gc {

JitEngine::JitEngine(std::unique_ptr<mlir::ExecutionEngine> eng)
    : engine(std::move(eng)) {}

JitEngine::JitEngine(std::unique_ptr<LLJIT> jit) : lljit(std::move(jit)) {}

JitEngine::~JitEngine() = default;

bool JitEngine::saveToFile(StringRef filename) {
  if (!engine) return false;
  engine->dumpToObjectFile(filename);
  return true;
}

std::unique_ptr<JitEngine>
JitEngine::loadFromFile(SmallString<128> &filePath,
                        ArrayRef<StringRef> sharedLibPaths) {
  if (!sys::fs::exists(filePath)) return nullptr;
  auto objBuf = MemoryBuffer::getFile(filePath, /*IsText=*/false,
                                      /*RequiresNullTerminator=*/false);
  if (!objBuf) return nullptr;

  auto jitOrErr = LLJITBuilder().create();
  if (!jitOrErr) {
    consumeError(jitOrErr.takeError());
    return nullptr;
  }
  auto &jit = *jitOrErr;

  if (!sharedLibPaths.empty()) {
    auto &mainJD = jit->getMainJITDylib();
    mainJD.addGenerator(
        cantFail(DynamicLibrarySearchGenerator::GetForCurrentProcess(
            jit->getDataLayout().getGlobalPrefix())));

    for (auto lib : sharedLibPaths) {
      auto gen = DynamicLibrarySearchGenerator::Load(
          lib.str().c_str(), jit->getDataLayout().getGlobalPrefix());
      if (gen) mainJD.addGenerator(std::move(*gen));
      else consumeError(gen.takeError());
    }
  }
  if (auto err = jit->addObjectFile(std::move(*objBuf))) {
    consumeError(std::move(err));
    return nullptr;
  }

  return std::make_unique<JitEngine>(std::move(jit));
}

Expected<void *> JitEngine::lookup(StringRef name) const {
  if (engine) return engine->lookup(name);
  auto sym = lljit->lookup(name);
  if (!sym) return sym.takeError();
  return sym->toPtr<void *>();
}

Expected<void (*)(void **)> JitEngine::lookupPacked(StringRef name) const {
  if (engine) return engine->lookupPacked(name);
  auto result = lookup("_mlir_" + name.str());
  if (!result) return result.takeError();
  return reinterpret_cast<void (*)(void **)>(*result);
}

using RegisterSymFn = SymbolMap (&)(MangleAndInterner);
template <> void JitEngine::registerSymbols<RegisterSymFn>(RegisterSymFn &&fn) {
  if (engine) {
    engine->registerSymbols(std::forward<RegisterSymFn>(fn));
    return;
  }
  auto &mainJD = lljit->getMainJITDylib();
  cantFail(mainJD.define(absoluteSymbols(fn(MangleAndInterner(
      mainJD.getExecutionSession(), lljit->getDataLayout())))));
}
} // namespace mlir::gc
