#include "Map.h"

#include <kernel/Context.h>

namespace gmac { namespace memory {

RegionMap Map::__global(paraver::mmGlobal);
RegionMap Map::__shared(paraver::mmShared);

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
    RegionMap(paraver::mmLocal)
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

}}
