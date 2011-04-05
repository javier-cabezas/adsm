#include "Slab.h"

#include "config/common.h"
#include "core/Mode.h"

namespace __impl { namespace memory { namespace allocator {

Cache &Slab::createCache(CacheMap &map, long_t key, size_t size)
{
    Cache *cache = new __impl::memory::allocator::Cache(size);
    map.insert(CacheMap::value_type(key, cache));
    return *cache;
}

Cache &Slab::get(core::Mode &current, long_t key, size_t size)
{
    ModeMap::iterator i;
    modes.lockRead();
    i = modes.find(&current);
    modes.unlock();
    if(i == modes.end()) {
        modes.lockWrite();
        Cache &ret = createCache(modes[&current], key, size);
        modes.unlock();
        return ret;
    }
    else {
        CacheMap::iterator j;
        j = i->second.find(key);
        if(j == i->second.end())
            return createCache(i->second, key, size);
        else
            return *j->second;
    }
}

void Slab::cleanup(core::Mode &current)
{
    ModeMap::iterator i;
    modes.lockRead();
    i = modes.find(&current);
    modes.unlock();
    if(i == modes.end()) return;
    CacheMap::iterator j;
    for(j = i->second.begin(); j != i->second.end(); j++) {
        delete j->second;
    }
    i->second.clear();
    modes.lockWrite();
    modes.erase(i);
    modes.unlock();
}

hostptr_t Slab::alloc(core::Mode &current, size_t size, hostptr_t addr)
{
    Cache &cache = get(current, long_t(addr) ^ size, size);
    TRACE(LOCAL,"Using cache %p", &cache);
    hostptr_t ret = cache.get();
    addresses.lockWrite();
    addresses.insert(AddressMap::value_type(ret, &cache));
    addresses.unlock();
    TRACE(LOCAL,"Retuning address %p", ret);
    return ret;
}

bool Slab::free(core::Mode &current, hostptr_t addr)
{
    addresses.lockWrite();
    AddressMap::iterator i = addresses.find(addr);
    if(i == addresses.end()) {
        addresses.unlock();
        TRACE(LOCAL,"%p was not delivered by slab allocator", addr); 
        return false;
    }
    TRACE(LOCAL,"Inserting %p in cache %p", addr, i->second);
    i->second->put(addr);
    addresses.erase(i);
    addresses.unlock();
    return true;
}

}}}
