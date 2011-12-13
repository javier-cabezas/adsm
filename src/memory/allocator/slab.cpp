#include "slab.h"

#include "config/common.h"
#include "core/address_space.h"

namespace __impl { namespace memory { namespace allocator {

cache &slab::createCache(core::address_space_ptr aspace, CacheMap &map, long_t key, size_t size)
{
    cache *cache = new __impl::memory::allocator::cache(manager_, aspace, size);
    std::pair<CacheMap::iterator, bool> ret = map.insert(CacheMap::value_type(key, cache));
    ASSERTION(ret.second == true);
    return *cache;
}

cache &slab::get(core::address_space_ptr current, long_t key, size_t size)
{
    CacheMap *map = NULL;
    map_aspace::iterator i;
    aspaces_.lock_read();
    i = aspaces_.find(current);
    if(i != aspaces_.end()) map = &(i->second);
    aspaces_.unlock();
    if(map == NULL) {
        aspaces_.lock_write();
        cache &ret = createCache(current, aspaces_[current], key, size);
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

void slab::cleanup(core::address_space_ptr current)
{
    map_aspace::iterator i;
    aspaces_.lock_read();
    i = aspaces_.find(current);
    aspaces_.unlock();
    if(i == aspaces_.end()) return;
    CacheMap::iterator j;
    for(j = i->second.begin(); j != i->second.end(); j++) {
        delete j->second;
    }
    i->second.clear();
    aspaces_.lock_write();
    aspaces_.erase(i);
    aspaces_.unlock();
}

hostptr_t slab::alloc(core::address_space_ptr current, size_t size, hostptr_t addr)
{
    cache &cache = get(current, long_t(addr), size);
    TRACE(LOCAL,"Using cache %p", &cache);
    hostptr_t ret = cache.get();
    if(ret == NULL) return NULL;
    addresses_.lock_write();
    addresses_.insert(map_address::value_type(ret, &cache));
    addresses_.unlock();
    TRACE(LOCAL,"Retuning address %p", ret);
    return ret;
}

bool slab::free(core::address_space_ptr current, hostptr_t addr)
{
    addresses_.lock_write();
    map_address::iterator i = addresses_.find(addr);
    if(i == addresses_.end()) {
        addresses_.unlock();
        TRACE(LOCAL,"%p was not delivered by slab allocator", addr); 
        return false;
    }
    TRACE(LOCAL,"Inserting %p in cache %p", addr, i->second);
    cache &cache = *(i->second);
    addresses_.erase(i);
    addresses_.unlock();
    cache.put(addr);
    return true;
}

}}}
