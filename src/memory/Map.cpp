#include "Map.h"

#include <kernel/Context.h>

namespace gmac { namespace memory {

//Map::SharedList *Map::__shared = NULL;
Map::__Map *Map::__global = NULL;
unsigned Map::count = 0;
MUTEX(Map::global);

void Map::clean()
{
	__Map::iterator i;
	for(i = __map.begin(); i != __map.end(); i++) {
		TRACE("Cleaning Region %p", i->second->start());
		globalLock();
		__global->erase(i->first);
		globalUnlock();
		delete i->second;
	}
	__map.clear();
}

Region *Map::remove(void *addr)
{
	__Map::iterator i;
	globalLock();
	i = __global->upper_bound(addr);
	assert(i != __global->end() && i->second->start() == addr);
	if(i->second->owner() == gmac::Context::current()) __global->erase(i);
	globalUnlock();
	// If the region is global (not owned by the context) return
	if(i->second->owner() != gmac::Context::current()) 
		return i->second;

	TRACE("Removing Region %p", i->second->start());
	i = __map.upper_bound(addr);
	assert(i != __map.end() && i->second->start() == addr);
	Region *ret = i->second;
	__map.erase(i);
	return ret;
}

}}
