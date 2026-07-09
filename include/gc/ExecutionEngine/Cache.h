//===-- Cache.h - Compiled binaries cache -----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cstring>
#include <optional>
#include <utility>

#include "gc/Utils/Log.h"
#include "gc/Utils/Misc.h"
#include "gc/Version.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Chrono.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SHA1.h"

#include "gc/ExecutionEngine/JitEngine.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

using namespace llvm;
using namespace mlir;

namespace mlir::gc::cache {
constexpr char GC_CACHE_BIN_FILE[] = "module.bin";
constexpr char GC_CACHE_INFO_FILE[] = "module.info";

struct CachedEngineInfo {
  uint64_t outArgsMask;
  bool isStatic;
  uint8_t argCount;
  char mainFuncName[128];

  CachedEngineInfo() = default;
  CachedEngineInfo(StringRef mainFuncName, bool isStatic, uint8_t argCount,
                   uint64_t outArgsMask)
      : outArgsMask(outArgsMask), isStatic(isStatic), argCount(argCount),
        mainFuncName{0} {
    assert(mainFuncName.size() < sizeof(this->mainFuncName));
    memcpy(this->mainFuncName, mainFuncName.data(), mainFuncName.size());
  }
};

inline bool isEnabled() {
  constexpr bool offByDefault =
#ifdef GC_CACHE_DEFAULT_OFF
      true;
#else
      false;
#endif
  static bool enabled = !misc::getEnv("GC_CACHE_OFF", offByDefault);
  return enabled;
}

inline SmallString<128> getCacheDir(ModuleOp mod, ArrayRef<uint8_t> salt = {}) {
  if (!isEnabled()) return {};

  static SmallString<128> cacheDir = [] {
    bool cleanParent = false;
    SmallString<128> path;
    if (auto env = misc::getEnv("GC_CACHE_DIR")) {
      path = env;
    } else {
      sys::path::cache_directory(path);
      sys::path::append(path, "intel");
      sys::path::append(path, "graph-compiler");
      sys::path::append(path, GC_VERSION_STRING);
      cleanParent = true;
    }

    auto len = path.size();
    sys::path::append(path, "cleanup.stamp");
    auto now = sys::TimePoint<>::clock::now();
    sys::fs::file_status stat;
    if (auto ec = sys::fs::status(path, stat);
        !ec &&
        now - stat.getLastModificationTime() < // Clean once a day by default
            std::chrono::hours(misc::getEnv("GC_CACHE_CLEAN_INTERVAL", 24))) {
      path.truncate(len);
      return path;
    }
    {
      std::error_code ec =
          sys::fs::create_directories(sys::path::parent_path(path));
      llvm::raw_fd_ostream os(path, ec);
      if (ec) gcLogE("Failed to update cleanup stamp ", path.c_str());
    }
    path.truncate(len);

    auto rmStaleDirs =
        [&now](StringRef root, std::chrono::hours staleTime,
               std::function<bool(sys::fs::directory_iterator)> predicate =
                   nullptr) {
          std::error_code ec;
          for (sys::fs::directory_iterator itr(root, ec), end;
               !ec && itr != end; itr.increment(ec)) {
            if (predicate && !predicate(itr)) continue;
            if (auto stat = itr->status();
                stat && (now - stat->getLastAccessedTime() > staleTime)) {
              auto path = itr->path();
              gcLogD("Removing stale cache folder ", path.c_str());
              if (sys::fs::remove_directories(path))
                gcLogE("Failed to remove stale cache folder ", path.c_str());
            }
          }
        };

    rmStaleDirs(
        path, std::chrono::hours(misc::getEnv("GC_CACHE_MAX_HOURS", 24 * 30)));
    if (cleanParent) // Clean cache remaining from other GC versions
      rmStaleDirs(
          sys::path::parent_path(path), std::chrono::hours(24),
          [&](sys::fs::directory_iterator itr) { return itr->path() != path; });

    return path;
  }();

  SHA1 key;
  std::string ir;
  llvm::raw_string_ostream os(ir);
  mod.print(os);
  key.update(GC_VERSION_STRING);
  key.update(ir);
  key.update(salt);
  key.final();
  SmallString<40> hash;
  toHex(key.result(), true, hash);
  SmallString<128> dir = cacheDir;
  sys::path::append(dir, hash);
  return dir;
}

inline std::optional<
    std::pair<std::unique_ptr<::gc::JitEngine>, CachedEngineInfo>>
load(SmallString<128> &cacheDir, ArrayRef<StringRef> sharedLibPaths = {}) {
  if (!isEnabled()) return {};

  SmallString<128> path = cacheDir;
  sys::path::append(path, GC_CACHE_INFO_FILE);
  if (!sys::fs::exists(path)) return std::nullopt;

  std::error_code ec;
  uint64_t size = 0;
  if (sys::fs::file_size(path, size) || size != sizeof(CachedEngineInfo))
    return std::nullopt;
  FILE *f = fopen(path.c_str(), "rb");
  if (!f) return std::nullopt;
  CachedEngineInfo info;
  bool ok = fread(&info, sizeof(info), 1, f) == 1;
  fclose(f);
  if (!ok) return std::nullopt;

  path.truncate(cacheDir.size());
  sys::path::append(path, GC_CACHE_BIN_FILE);
  auto eng = JitEngine::loadFromFile(path, sharedLibPaths);
  if (!eng) return std::nullopt;

  gcLogD("Loaded compiled module from cache: ", path.c_str());
  return std::make_pair(std::move(eng), info);
}

inline void save(SmallString<128> &cacheDir, JitEngine *engine,
                 CachedEngineInfo &info,
                 SmallDenseMap<StringRef, ModuleOp> &dumpModules) {
  if (!isEnabled()) return;

  if (std::error_code ec = sys::fs::create_directories(cacheDir, true)) {
    gcLogE("Failed to create cache directory ", cacheDir.c_str(), ": ",
           ec.message());
    return;
  }

  SmallString<128> path = cacheDir;

  {
    std::error_code ec;
    sys::path::append(path, GC_CACHE_INFO_FILE);
    llvm::raw_fd_ostream os(path, ec);
    if (ec) {
      gcLogE("Failed to create the cache file `", path.c_str(),
             "`: ", ec.message());
      return;
    }
    os.write(reinterpret_cast<const char *>(&info), sizeof(info));
  }

  path.truncate(cacheDir.size());
  sys::path::append(path, GC_CACHE_BIN_FILE);
  if (!engine->saveToFile(StringRef(path))) {
    gcLogE("Failed to dump engine to cache file `", path.c_str(), "`.");
    return;
  }
  gcLogD("Saved compiled module to cache: ", cacheDir.c_str());

  for (auto &[name, mod] : dumpModules) {
    path.truncate(cacheDir.size());
    sys::path::append(path, name);
    std::error_code ec;
    llvm::raw_fd_ostream os(path, ec);
    if (ec) {
      gcLogE("Failed to create the cache file `", path.c_str(),
             "`: ", ec.message());
      continue;
    }
    mod->print(os, OpPrintingFlags().useLocalScope());
  }
}
} // namespace mlir::gc::cache