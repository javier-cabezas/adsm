#include "Map.h"

#include <kernel/Context.h>

namespace gmac { namespace memory {

RegionMap *Map::__global = NULL;
unsigned Map::count = 0;
gmac::util::RWLock Map::global(paraver::mmGlobal);

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
    global.lockRead();
    i = __global->upper_bound(addr);
    if(i != __global->end() && i->second->start() <= addr)
        ret = i->second;
    global.unlock();
    return ret;
}

void
Map::clean()
{
	RegionMap::iterator i;
	lockWrite();
	for(i = begin(); i != end(); i++) {
		TRACE("Cleaning Region %p", i->second->start());
		global.lockWrite();
		__global->erase(i->first);
		global.unlock();
		delete i->second;
	}
	clear();
	unlock();
}

Map::Map() :
    util::RWLock(paraver::mmLocal)
{
    global.lockWrite();
    if(__global == NULL) __global = new RegionMap();
    count++;
    global.unlock();
}

Map::~Map()
{
    TRACE("Cleaning Memory Map");
    clean();
    global.lockWrite();
    count--;
    if(count == 0) {
        delete __global;
        __global = NULL;
    }
    global.unlock();
}

void
Map::init()
{}

Region *Map::remove(void *addr)
{
    RegionMap::iterator i;
    global.lockWrite();
    i = __global->upper_bound(addr);
    Region * r = i->second;
    assert(i != __global->end() && r->start() == addr);
    Context * ctx = Context::current();
    if(r->owner() == ctx) __global->erase(i);
    global.unlock();
    // If the region is global (not owned by the context) return
    if(r->owner() != ctx)
        return r;

    TRACE("Removing Region %p", r->start());
    lockWrite();
    i = upper_bound(addr);
    assert(i != end() && r->start() == addr);
    erase(i);
    unlock();
    return r;
}

}}
