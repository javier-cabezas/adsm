#include "Slab.h"

#include <kernel/Mode.h>

namespace gmac { namespace memory { namespace allocator {

Cache &Slab::createCache(CacheMap &map, long key, size_t size)
{
    Cache *cache = new Cache(size);
    map.insert(CacheMap::value_type(key, cache));
    return *cache;
}

Cache &Slab::get(long key, size_t size)
{
    Cache *cache = NULL;
    ModeMap::iterator i;
    modes.lockRead();
    i = modes.find(Mode::current());
    modes.unlock();
    if(i == modes.end()) {
        modes.lockWrite();
        Cache &ret = createCache(modes[Mode::current()], key, size);
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

void Slab::cleanup()
{
    ModeMap::iterator i;
    modes.lockRead();
    i = modes.find(Mode::current());
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

void *Slab::alloc(size_t size, void *addr)
{
    Cache &cache = get((unsigned long)addr, size);
    trace("Using cache %p", &cache);
    void *ret = cache.get();
    addresses.lockWrite();
    addresses.insert(AddressMap::value_type(ret, &cache));
    addresses.unlock();
    trace("Retuning address %p", ret);
    return ret;
}

bool Slab::free(void *addr)
{
    addresses.lockRead();
    AddressMap::iterator i = addresses.find(addr);
    if(i == addresses.end()) {
        trace("%p was not delivered by slab allocator", addr); 
        return false;
    }
    addresses.unlock();
    i->second->put(addr);
    return true;
}

}}}
