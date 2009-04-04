#include "CacheRegion.h"
#include "CacheManager.h"

#include <memory/os/Memory.h>

#include <algorithm>

namespace gmac {
CacheRegion::CacheRegion(CacheManager &manager, void *addr, size_t size,
		size_t cacheLine) :
	ProtRegion(addr, size),
	manager(manager),
	cacheLine(cacheLine),
	offset((unsigned long)addr & (cacheLine -1))
{
	TRACE("CacheRegion Starts");
	for(size_t s = 0; s < size; s += cacheLine) {
		void *p = (void *)((uint8_t *)addr + s);
		size_t regionSize = ((size -s) > cacheLine) ? cacheLine : (size - s);
		TRACE("New SubRegion(0x%x) %p (%d bytes)", s, addr, regionSize);
		ProtSubRegion *region = new ProtSubRegion(this, p, regionSize);
		set[p] = region;
		memory.push_back(region);
	}
	TRACE("CacheRegion Ends");
}

CacheRegion::~CacheRegion()
{
	Set::const_iterator i;
	for(i = set.begin(); i != set.end(); i++)
		delete i->second;
	set.clear();
}

void CacheRegion::invalidate()
{
	// Check if the region is already invalid
	if(memory.empty()) return;

	// Protect the region
	Memory::protect(__void(addr), size, PROT_NONE);
	// Invalidate those sub-regions that are present in memory
	List::iterator i;
	for(i = memory.begin(); i != memory.end(); i++) {
		(*i)->silentInvalidate();
	}
	memory.clear();
}


};
