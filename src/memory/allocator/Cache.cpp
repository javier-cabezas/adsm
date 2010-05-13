#include "Cache.h"

#include <kernel/Context.h>
#include <memory/Manager.h>

#include <util/Parameter.h>
#include <util/Private.h>

#include <debug.h>

namespace gmac { namespace memory { namespace allocator {

Arena::Arena(Manager *manager, size_t objSize) :
    ptr(NULL),
    size(0),
    manager(manager)
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
    Arena *arena = new Arena(manager, objectSize);
    arenas.insert(ArenaMap::value_type(arena->address(), arena));
    return arena->get();
}

}}}
