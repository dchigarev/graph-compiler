//===-- Cache.cpp - Compiled binaries cache ---------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstring>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <utility>

#include "gc/ExecutionEngine/Cache.h"
#include "gc/Utils/Error.h"
#include "gc/Utils/Log.h"
#include "gc/Utils/Misc.h"
#include "gc/Version.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/Support/Chrono.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SHA1.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

using namespace llvm;
using namespace llvm::orc;
using namespace mlir;

namespace llvm {
template <> struct DenseMapInfo<gc::cache::Key::MapKey> {
  static gc::cache::Key::MapKey getEmptyKey() { return {{}, 0}; }
  static gc::cache::Key::MapKey getTombstoneKey() { return {{}, 2}; }
  static unsigned getHashValue(const gc::cache::Key::MapKey &key) {
    return hash_combine_range(key.hash.begin(), key.hash.end());
  }
  static bool isEqual(const gc::cache::Key::MapKey &lhs,
                      const gc::cache::Key::MapKey &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

namespace {
struct CacheInfo {
  uint64_t outArgsMask;
  bool isCtxMain;
  uint8_t argCount;
  char mainFuncName[128];

  CacheInfo() = default;
  CacheInfo(StringRef mainFuncName, bool isCtxMain, uint8_t argCount,
            uint64_t outArgsMask)
      : outArgsMask(outArgsMask), isCtxMain(isCtxMain), argCount(argCount),
        mainFuncName{0} {
    assert(mainFuncName.size() < sizeof(this->mainFuncName));
    memcpy(this->mainFuncName, mainFuncName.data(), mainFuncName.size());
  }
};

SmallString<128> getCacheDir(gc::cache::Key &key) {
  if (!gc::cache::isFileCacheEnabled()) return {};

  static SmallString<128> cacheDir = []() {
    bool cleanParent = false;
    SmallString<128> path;
    if (auto env = gc::misc::getEnv("GC_CACHE_DIR")) {
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
            std::chrono::hours(
                gc::misc::getEnv("GC_CACHE_CLEAN_INTERVAL", 24))) {
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

    rmStaleDirs(path, std::chrono::hours(
                          gc::misc::getEnv("GC_CACHE_MAX_HOURS", 24 * 30)));
    if (cleanParent) // Clean cache remaining from other GC versions
      rmStaleDirs(
          sys::path::parent_path(path), std::chrono::hours(24),
          [&](sys::fs::directory_iterator itr) { return itr->path() != path; });

    return path;
  }();

  SmallString<128> dir = cacheDir;
  sys::path::append(dir, key.hex());
  return dir;
}

llvm::Expected<std::variant<gc::cache::CachedEngine::WrappedMainFunc,
                            gc::cache::CachedEngine::CtxMainFunc>>
findMainFunc(std::unique_ptr<::gc::JitEngine> &eng, StringRef mainFuncName,
             bool isCtxMain) {
  if (isCtxMain) {
    auto expect = eng->lookup(mainFuncName);
    if (!expect)
      return gcMakeErr("Compiled function '", mainFuncName.begin(),
                       "' not found!");
    return reinterpret_cast<gc::cache::CachedEngine::CtxMainFunc>(*expect);
  } else {
    auto expect = eng->lookupPacked(mainFuncName);
    if (!expect)
      return gcMakeErr("Packed function '", mainFuncName.begin(),
                       "' not found!");
    return *expect;
  }
}
} // namespace

namespace mlir::gc::cache {
constexpr char GC_CACHE_BIN_FILE[] = "module.bin";
constexpr char GC_CACHE_INFO_FILE[] = "module.info";

struct Cache {
  static std::shared_ptr<const CachedEngine> get(const Key &key) {
    auto &instance = Cache::instance();
    std::shared_lock<std::shared_mutex> lock(instance.mux);
    auto it = instance.map.find(key.mapKey);
    if (it == instance.map.end()) return nullptr;
    return it->second;
  }

  static std::shared_ptr<const CachedEngine>
  put(const Key &key, std::unique_ptr<const CachedEngine> engine) {
    auto &instance = Cache::instance();
    std::unique_lock<std::shared_mutex> lock(instance.mux);
    auto [it, inserted] =
        instance.map.try_emplace(key.mapKey, std::move(engine));
    return it->second;
  }

private:
  std::shared_mutex mux{};
  llvm::DenseMap<Key::MapKey, std::shared_ptr<const CachedEngine>> map;

  static Cache &instance() {
    // Intentionally never destroyed: destroying it at exit would tear down
    // the JIT engines it owns after LLVM's own global codegen state may
    // already be gone, causing crashes during static destruction.
    static Cache *cache = new Cache();
    return *cache;
  }
};

SmallString<40> &Key::hex() {
  if (hexHash.empty()) toHex(mapKey.hash, true, hexHash);
  return hexHash;
}

using RegisterSymFn = SymbolMap (&)(MangleAndInterner);
template <>
std::shared_ptr<const CachedEngine> load(Key &key,
                                         RegisterSymFn &&registerSymbols,
                                         void (*destructor)(::gc::JitEngine &),
                                         ArrayRef<StringRef> sharedLibPaths) {
  auto cached = Cache::get(key);
  if (cached) return cached;
  if (!isFileCacheEnabled()) return nullptr;

  auto cacheDir = getCacheDir(key);
  SmallString<128> path = cacheDir;
  sys::path::append(path, GC_CACHE_INFO_FILE);
  if (!sys::fs::exists(path)) return nullptr;

  std::error_code ec;
  uint64_t size = 0;
  if (sys::fs::file_size(path, size) || size != sizeof(CacheInfo))
    return nullptr;
  FILE *f = fopen(path.c_str(), "rb");
  if (!f) return nullptr;
  CacheInfo info;
  bool ok = fread(&info, sizeof(info), 1, f) == 1;
  fclose(f);
  if (!ok) return nullptr;

  path.truncate(cacheDir.size());
  sys::path::append(path, GC_CACHE_BIN_FILE);
  auto eng = JitEngine::loadFromFile(path, sharedLibPaths);
  if (!eng) return nullptr;
  eng->registerSymbols(std::forward<RegisterSymFn>(registerSymbols));

  gcLogD("Loaded compiled module from cache: ", path.c_str());
  auto mainFunc = findMainFunc(eng, info.mainFuncName, info.isCtxMain);
  if (!mainFunc) {
    llvm::consumeError(mainFunc.takeError());
    return nullptr;
  }
  return Cache::put(key, std::make_unique<const CachedEngine>(
                             std::move(eng), *mainFunc, info.outArgsMask,
                             info.argCount, destructor));
}

llvm::Expected<std::shared_ptr<const CachedEngine>>
save(Key &key, std::unique_ptr<ExecutionEngine> engine, StringRef mainFuncName,
     bool isCtxMain, uint64_t outArgsMask, uint8_t argCount,
     SmallDenseMap<StringRef, std::variant<ModuleOp, StringAttr>> &dumpModules,
     void (*destructor)(::gc::JitEngine &)) {
  auto eng = std::make_unique<::gc::JitEngine>(std::move(engine));
  auto mainFunc = findMainFunc(eng, mainFuncName, isCtxMain);
  if (!mainFunc) return mainFunc.takeError();
  auto cached = Cache::put(
      key, std::make_unique<const CachedEngine>(
               std::move(eng), *mainFunc, outArgsMask, argCount, destructor));
  if (!isFileCacheEnabled()) return cached;

  SmallString<128> cacheDir = getCacheDir(key);
  if (std::error_code ec = sys::fs::create_directories(cacheDir, true)) {
    gcLogE("Failed to create cache directory ", cacheDir.c_str(), ": ",
           ec.message());
    return cached;
  }

  SmallString<128> path = cacheDir;

  {
    std::error_code ec;
    sys::path::append(path, GC_CACHE_INFO_FILE);
    llvm::raw_fd_ostream os(path, ec);
    if (ec) {
      gcLogE("Failed to create the cache file `", path.c_str(),
             "`: ", ec.message());
      return cached;
    }
    CacheInfo info{mainFuncName.str(), cached->isCtxMain(), cached->argCount,
                   cached->outArgsMask};
    os.write(reinterpret_cast<const char *>(&info), sizeof(info));
  }

  path.truncate(cacheDir.size());
  sys::path::append(path, GC_CACHE_BIN_FILE);
  if (!cached->engine->saveToFile(StringRef(path))) {
    gcLogE("Failed to dump engine to cache file `", path.c_str(), "`.");
    return cached;
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
    if (auto *m = std::get_if<ModuleOp>(&mod)) {
      (*m)->print(os, OpPrintingFlags().useLocalScope());
    } else if (auto *s = std::get_if<StringAttr>(&mod)) {
      os << s->getValue();
    }
  }
  return cached;
}
} // namespace mlir::gc::cache
