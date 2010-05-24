#include "Slab.h"

namespace gmac { namespace memory { namespace allocator {

Cache &Slab::createCache(CacheMap &map, long key, size_t size)
{
    Cache *cache = new Cache(manager, size);
    map.insert(CacheMap::value_type(key, cache));
    return *cache;
}

Cache &Slab::get(long key, size_t size)
{
    Cache *cache = NULL;
    ContextMap::iterator i;
    i = contexts.find(Context::current());
    if(i == contexts.end()) {
        return createCache(contexts[Context::current()], key, size);
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
    ContextMap::iterator i;
    i = contexts.find(Context::current());
    if(i == contexts.end()) return;
    CacheMap::iterator j;
    for(j = i->second.begin(); j != i->second.end(); j++) {
        delete j->second;
    }
    i->second.clear();
    contexts.erase(i);
    
}

void *Slab::alloc(size_t size, void *addr)
{
    Cache &cache = get((unsigned long)addr, size);
    trace("Using cache %p", &cache);
    void *ret = cache.get();
    addresses.insert(AddressMap::value_type(ret, &cache));
    return ret;
}

bool Slab::free(void *addr)
{
    AddressMap::iterator i = addresses.find(addr);
    if(i == addresses.end()) {
        trace("%p was not delivered by slab allocator", addr); 
        return false;
    }
    i->second->put(addr);
    return true;
}

}}}
