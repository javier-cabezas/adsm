#include "Map.h"

#include <kernel/Process.h>
#include <kernel/Context.h>

namespace gmac { namespace memory {

RegionMap::RegionMap(paraver::LockName name) :
    RWLock(name)
{
}

Region *
Map::localFind(const void *addr)
{
    RegionMap::const_iterator i;
    Region *ret = NULL;
    i = upper_bound(addr);
    if(i != end() && i->second->start() <= addr) {
        ret = i->second;
    }
    return ret;
}

Region *
Map::globalFind(const void *addr)
{
    RegionMap::const_iterator i;
    Region *ret = NULL;
    RegionMap &__global = proc->global();
    __global.lockRead();
    i = __global.upper_bound(addr);
    if(i != __global.end() && i->second->start() <= addr)
        ret = i->second;
    __global.unlock();
    return ret;
}

Region *
Map::sharedFind(const void *addr)
{
    RegionMap::const_iterator i;
    Region *ret = NULL;
    RegionMap &__shared = proc->shared();
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
		TRACE("Cleaning Region %p", i->second->start());
		__global.lockWrite();
		__global.erase(i->first);
		__global.unlock();
		delete i->second;
	}
	clear();
	unlock();
}

Map::Map() :
    RegionMap(LockMmLocal)
{
}

Map::~Map()
{
    TRACE("Cleaning Memory Map");
    clean();
}

void
Map::init()
{}

Region *Map::remove(void *addr)
{
    RegionMap::iterator i;
    RegionMap &__global = proc->global();
    __global.lockWrite();
    i = __global.upper_bound(addr);
    Region * r = i->second;
    ASSERT(i != __global.end() && r->start() == addr);
    Context * ctx = Context::current();
    if(r->owner() == ctx) __global.erase(i);
    __global.unlock();
    // If the region is global (not owned by the context) return
    if(r->owner() != ctx)
        return r;

    TRACE("Removing Region %p", r->start());
    lockWrite();
    i = upper_bound(addr);
    ASSERT(i != end() && r->start() == addr);
    erase(i);
    unlock();
    return r;
}

void Map::insert(Region *r)
{
    lockWrite();
    RegionMap::insert(value_type(r->end(), r));
    unlock();

    RegionMap &__global = proc->global();
    __global.lockWrite();
    __global.insert(value_type(r->end(), r));
    __global.unlock();
}

void Map::addShared(Region * r)
{
    RegionMap &__shared = proc->shared();
    __shared.lockWrite();
    __shared.insert(value_type(r->end(), r));
    __shared.unlock();
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
}

bool Map::isShared(const void *addr)
{
    bool ret;
    RegionMap &__shared = proc->shared();
    __shared.lockRead();
    ret = __shared.find(addr) != __shared.end();
    __shared.unlock();

    return ret;
}

RegionMap & Map::shared()
{
    return proc->shared();
}


}}
