#include "Slab.h"

#include "core/Mode.h"

namespace __impl { namespace memory { namespace allocator {

Cache &Slab::createCache(CacheMap &map, long key, size_t size)
{
    Cache *cache = new __impl::memory::allocator::Cache(size);
    map.insert(CacheMap::value_type(key, cache));
    return *cache;
}

Cache &Slab::get(long key, size_t size)
{
    ModeMap::iterator i;
    modes.lockRead();
    core::Mode *mode = &core::Mode::current();
    i = modes.find(mode);
    modes.unlock();
    if(i == modes.end()) {
        modes.lockWrite();
        Cache &ret = createCache(modes[mode], key, size);
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
    i = modes.find(&core::Mode::current());
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
    Cache &cache = get((unsigned long)addr ^ (unsigned long)size, size);
    TRACE(LOCAL,"Using cache %p", &cache);
    void *ret = cache.get();
    addresses.lockWrite();
    addresses.insert(AddressMap::value_type(ret, &cache));
    addresses.unlock();
    TRACE(LOCAL,"Retuning address %p", ret);
    return ret;
}

bool Slab::free(void *addr)
{
    addresses.lockRead();
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
