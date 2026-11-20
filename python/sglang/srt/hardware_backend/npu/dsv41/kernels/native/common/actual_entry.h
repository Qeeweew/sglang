#pragma once
#include "jit.h"
#include <array>
#include <map>
#include <memory>

namespace native_jit {
// Compile configurations observed by real prepare calls. The last-entry cache
// keeps a repeated warm call free of map traversal/string construction. Keys
// contain scalar geometry, finite storage/dataflow policy and prepared worker budget;
// addresses, payloads and live lengths are excluded.
template <class Geometry> class KernelCache {
  public:
    using Key = Geometry;

  private:
    std::map<Key, std::unique_ptr<Kernel>> entries_;
    Key last_key_{};
    Kernel *last_ = nullptr;

  public:
    template <class Factory> Kernel &get(const Key &key, Factory factory) {
        if (last_ && last_key_ == key)
            return *last_;
        auto iterator = entries_.find(key);
        if (iterator == entries_.end())
            iterator = entries_.emplace(key, factory()).first;
        last_key_ = key;
        last_ = iterator->second.get();
        return *last_;
    }
};

template <size_t Dimensions> using ActualEntries = KernelCache<std::array<uint64_t, Dimensions>>;
} // namespace native_jit
