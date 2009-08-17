#include "RollingRegion.h"
#include "RollingManager.h"

#include "os/Memory.h"

#include <algorithm>

namespace gmac {
RollingRegion::RollingRegion(RollingManager &manager, void *addr, size_t size,
		size_t cacheLine) :
	ProtRegion(addr, size),
	manager(manager),
	cacheLine(cacheLine),
	offset((unsigned long)addr & (cacheLine -1))
{
	TRACE("RollingRegion Starts");
	for(size_t s = 0; s < size; s += cacheLine) {
		void *p = (void *)((uint8_t *)addr + s);
		size_t regionSize = ((size -s) > cacheLine) ? cacheLine : (size - s);
		ProtSubRegion *region = new ProtSubRegion(this, p, regionSize);
		TRACE("New SubRegion(0x%x) %p (%d bytes) [%p]", s, addr, regionSize, region);
		set[p] = region;
		memory.push_back(region);
	}
	TRACE("RollingRegion Ends");
}

RollingRegion::~RollingRegion()
{
	Set::const_iterator i;
	for(i = set.begin(); i != set.end(); i++) {
		manager.invalidate(i->second);
		delete i->second;
	}
	set.clear();
}


void RollingRegion::invalidate()
{
	TRACE("RollingRegion Invalidate %p (%d bytes)", addr, size);
	// Check if the region is already invalid
	if(memory.empty()) return;

	// Protect the region
	Memory::protect(__void(addr), size, PROT_NONE);
	// Invalidate those sub-regions that are present in memory
	List::iterator i;
	for(i = memory.begin(); i != memory.end(); i++) {
		TRACE("Invalidate SubRegion %p (%d bytes)", (*i)->getAddress(),
				(*i)->getSize());
		(*i)->silentInvalidate();
	}
	memory.clear();
}

void RollingRegion::dirty()
{
	// Check if the region is already invalid
	if(memory.empty()) return;

	// Invalidate those sub-regions that are present in memory
	List::iterator i;
	for(i = memory.begin(); i != memory.end(); i++) {
		TRACE("Dirty call forces write");
		manager.write(*i, (*i)->getAddress());
	}
}

};
