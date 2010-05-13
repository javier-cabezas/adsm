#include "Cache.h"

#include <gmac/init.h>
#include <kernel/Context.h>
#include <memory/Manager.h>

#include <util/Parameter.h>
#include <util/Private.h>

#include <debug.h>

namespace gmac { namespace memory { namespace allocator {

Cache::ContextMap Cache::map;

Arena::Arena(size_t objSize) :
    ptr(NULL),
    size(0)
{
    Context *ctx = Context::current();
    ctx->lockRead();
    gmacError_t ret = manager->malloc(ctx, &ptr, paramPageSize);
    ctx->unlock();
    if(ret != gmacSuccess) return;
    for(size_t s = 0; s < paramPageSize; s += objSize, size++)
        __objects.push_back((void *)((uint8_t *)ptr + objSize));
}

Arena::~Arena()
{
    CFATAL(__objects.size() == size, "Destroying non-full Arena");
    __objects.clear();
    Context *ctx = Context::current();
    ctx->lockRead();
    gmacError_t ret = manager->free(ctx, ptr);
    ctx->unlock();
}

Cache::Cache(long key, size_t size) :
    objectSize(size),
    arenaSize(paramPageSize)
{
    map[Context::current()].insert(CacheMap::value_type(key, this));
}

Cache::~Cache()
{
    ArenaMap::iterator i;
    for(i = arenas.begin(); i != arenas.end(); i++) {
        delete i->second;
    }
  
}

void *Cache::get()
{
    ArenaMap::iterator i;
    for(i = arenas.begin(); i != arenas.end(); i++) {
        if(i->second->empty()) continue;
        return i->second->get();
    }
    // There are no free objects in any arena
    Arena *arena = new Arena(objectSize);
    arenas.insert(ArenaMap::value_type(arena->address(), arena));
    return arena->get();
}

Cache &Cache::get(long key, size_t size)
{
    Cache *cache = NULL;
    ContextMap::iterator i;
    i = map.find(Context::current());
    if(i == map.end()) cache = new Cache(key, size);
    else {
        CacheMap::iterator j;
        j = i->second.find(key);
        if(j == i->second.end()) cache = new Cache(key, size);
        else cache = j->second;
    }
    return *cache;
}

void Cache::cleanup()
{
    ContextMap::iterator i;
    i = map.find(Context::current());
    if(i == map.end()) return;
    CacheMap::iterator j;
    for(j = i->second.begin(); j != i->second.end(); j++)
        delete j->second;
    i->second.clear();
    map.erase(i);
}

}}}
