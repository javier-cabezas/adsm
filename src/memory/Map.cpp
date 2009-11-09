#include "Map.h"

namespace gmac { namespace memory {

Map::__Map *Map::__global = NULL;
unsigned Map::count = 0;
MUTEX(Map::global);

Region *Map::remove(void *addr)
{
	__Map::iterator i;
	globalLock();
	i = __map.upper_bound(addr);
	assert(i != __map.end() && i->second->start() == addr);
	Region *ret = i->second;
	__map.erase(i);
	i = __global->upper_bound(addr);
	assert(i != __global->end() && i->second->start() == addr);
	__global->erase(i);
	globalUnlock();
	return ret;
}

}}
