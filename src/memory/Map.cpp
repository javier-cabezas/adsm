#include "Map.h"

#include <kernel/Context.h>

namespace gmac { namespace memory {

Map::__Map *Map::__global = NULL;
unsigned Map::count = 0;
util::RWLock Map::global(paraver::mmGlobal);

Region *
Map::localFind(const void *addr)
{
    __Map::const_iterator i;
    Region *ret = NULL;
    i = __map.upper_bound(addr);
    if(i != __map.end() && i->second->start() <= addr) {
        ret = i->second;
    }
    return ret;
}

Region *
Map::globalFind(const void *addr)
{
    __Map::const_iterator i;
    Region *ret = NULL;
    i = __global->upper_bound(addr);
    if(i != __global->end() && i->second->start() <= addr)
        ret = i->second;
    return ret;
}

void
Map::clean()
{
	__Map::iterator i;
	local.write();
	for(i = __map.begin(); i != __map.end(); i++) {
		TRACE("Cleaning Region %p", i->second->start());
		global.write();
		__global->erase(i->first);
		global.unlock();
		delete i->second;
	}
	__map.clear();
	local.unlock();
}

Map::Map() :
    local(paraver::mmLocal)
{
    global.write();
    if(__global == NULL) __global = new __Map();
    count++;
    global.unlock();
}

Map::~Map()
{
    TRACE("Cleaning Memory Map");
    clean();
    global.write();
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
	__Map::iterator i;
	global.write();
	i = __global->upper_bound(addr);
	assert(i != __global->end() && i->second->start() == addr);
    Region *region = i->second;
    bool self = (region->owner() == Context::current());
    if(self == true) __global->erase(i);
	global.unlock();
	// If the region is global (not owned by the context) return
    if(self == false) return region;

	TRACE("Removing Region %p", region->start());
	local.write();
	i = __map.upper_bound(addr);
	assert(i != __map.end() && i->second->start() == addr);
	Region *ret = i->second;
	__map.erase(i);
	local.unlock();
	return ret;
}

}}
