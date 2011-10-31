#include "Slab.h"

#include "config/common.h"
#include "core/address_space.h"

namespace __impl { namespace memory { namespace allocator {

Cache &Slab::createCache(core::address_space &aspace, CacheMap &map, long_t key, size_t size)
{
    Cache *cache = new __impl::memory::allocator::Cache(manager_, aspace, size);
    std::pair<CacheMap::iterator, bool> ret = map.insert(CacheMap::value_type(key, cache));
    ASSERTION(ret.second == true);
    return *cache;
}

Cache &Slab::get(core::address_space &current, long_t key, size_t size)
{
    CacheMap *map = NULL;
    aspace_map::iterator i;
    aspaces_.lockRead();
    i = aspaces_.find(&current);
    if(i != aspaces_.end()) map = &(i->second);
    aspaces_.unlock();
    if(map == NULL) {
        aspaces_.lockWrite();
        Cache &ret = createCache(current, aspaces_[&current], key, size);
        aspaces_.unlock();
        return ret;
    }
    else {
        CacheMap::iterator j;
        j = map->find(key);
        if(j == map->end())
            return createCache(current, *map, key, size);
        else
            return *j->second;
    }
}

void Slab::cleanup(core::address_space &current)
{
    aspace_map::iterator i;
    aspaces_.lockRead();
    i = aspaces_.find(&current);
    aspaces_.unlock();
    if(i == aspaces_.end()) return;
    CacheMap::iterator j;
    for(j = i->second.begin(); j != i->second.end(); j++) {
        delete j->second;
    }
    i->second.clear();
    aspaces_.lockWrite();
    aspaces_.erase(i);
    aspaces_.unlock();
}

hostptr_t Slab::alloc(core::address_space &current, size_t size, hostptr_t addr)
{
    Cache &cache = get(current, long_t(addr), size);
    TRACE(LOCAL,"Using cache %p", &cache);
    hostptr_t ret = cache.get();
    if(ret == NULL) return NULL;
    addresses_.lockWrite();
    addresses_.insert(AddressMap::value_type(ret, &cache));
    addresses_.unlock();
    TRACE(LOCAL,"Retuning address %p", ret);
    return ret;
}

bool Slab::free(core::address_space &current, hostptr_t addr)
{
    addresses_.lockWrite();
    AddressMap::iterator i = addresses_.find(addr);
    if(i == addresses_.end()) {
        addresses_.unlock();
        TRACE(LOCAL,"%p was not delivered by slab allocator", addr); 
        return false;
    }
    TRACE(LOCAL,"Inserting %p in cache %p", addr, i->second);
    Cache &cache = *(i->second);
    addresses_.erase(i);
    addresses_.unlock();
    cache.put(addr);
    return true;
}

}}}
