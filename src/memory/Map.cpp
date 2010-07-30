#include "Map.h"

#include <kernel/Process.h>
#include <kernel/Context.h>

#include <gmac/init.h>

namespace gmac { namespace memory {


Object *
Map::localFind(const void *addr)
{
    ObjectMap::const_iterator i;
    Object *ret = NULL;
    i = upper_bound(addr);
    if(i != end() && i->second->start() <= addr) {
        ret = i->second;
    }
    return ret;
}

Object *
Map::globalFind(const void *addr)
{
    ObjectMap::const_iterator i;
    Object *ret = NULL;
    ObjectMap &__global = proc->global();
    __global.lockRead();
    i = __global.upper_bound(addr);
    if(i != __global.end() && i->second->start() <= addr)
        ret = i->second;
    __global.unlock();
    return ret;
}

Object *
Map::sharedFind(const void *addr)
{
    ObjectMap::const_iterator i;
    Object *ret = NULL;
    ObjectMap &__shared = proc->shared();
    __shared.lockRead();
    i = __shared.upper_bound(addr);
    if(i != __shared.end() && i->second->start() <= addr)
        ret = i->second;
    __shared.unlock();
    return ret;
}

void
Map::clean()
{
	RegionMap::iterator i;
    RegionMap &__global = proc->global();
	lockWrite();
	for(i = begin(); i != end(); i++) {
		trace("Cleaning Object %p", i->second->start());
		__global.lockWrite();
		__global.erase(i->first);
		__global.unlock();
		delete i->second;
	}
	clear();
	unlock();
}


void Map::insert(Object *obj)
{
    ObjectMap::insert(value_type(obj->end(), obj));

    ObjectMap &__global = proc->global();
    __global.lockWrite();
    __global.insert(value_type(obj->end(), obj));
    __global.unlock();
}


Object *Map::remove(const void *addr)
{
    ObjectMap::iterator i;
    ObjectMap &__global = proc->global();
    __global.lockWrite();
    i = __global.upper_bound(addr);
    Object *obj = i->second;
    assertion(i != __global.end() && obj->start() == addr);
    assertion(obj->owner() == Context::current());
    __global.unlock();
    // If the region is global (not owned by the context) return

    trace("Removing Object %p", obj->start());
    i = upper_bound(addr);
    assertion(i != end() && obj->start() == addr);
    erase(i);
    return obj;
}



void Map::insertShared(Region * r)
{
    ObjectMap &__shared = proc->shared();
    __shared.lockWrite();
    __shared.insert(value_type(r->end(), r));
    __shared.unlock();
    util::Logger::TRACE("Added shared region @ %p", r->start());
}

void Map::removeShared(Region * r)
{
    Map::iterator i;
    RegionMap &__shared = proc->shared();
    __shared.lockWrite();
    for (i = __shared.begin(); i != __shared.end(); i++) {
        if (r == i->second) {
            __shared.erase(i);
            break;
        }
    }
    __shared.unlock();
    util::Logger::TRACE("Removed shared region @ %p", r->start());
}

}}
