#include "MemRegion.h"
#include "debug.h"

#include <string.h>

#include <algorithm>

namespace gmac {
std::list<ProtRegion *> ProtRegion::regionList;
MUTEX(ProtRegion::regionMutex);

ProtRegion::ProtRegion(MemHandler &memHandler, void *addr, size_t size) :
	MemRegion(addr, size),
	memHandler(memHandler),
	dirty(false)
{
	if(regionList.empty()) {
		MUTEX_INIT(regionMutex);
		setHandler();
	}
	MUTEX_LOCK(regionMutex);
	regionList.push_front(this);
	MUTEX_UNLOCK(regionMutex);
	TRACE("New ProtRegion %p (%d bytes)", addr, size);
}

ProtRegion::~ProtRegion()
{
	std::list<ProtRegion *>::iterator i;
	MUTEX_LOCK(regionMutex);
	i = std::find(regionList.begin(), regionList.end(), this);
	if(i != regionList.end()) regionList.erase(i);
	if(regionList.empty()) restoreHandler();
	MUTEX_UNLOCK(regionMutex);
}

};
