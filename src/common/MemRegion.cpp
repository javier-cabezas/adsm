#include "MemRegion.h"
#include "debug.h"

#include <string.h>

#include <algorithm>

namespace gmac {
std::list<ProtRegion *> ProtRegion::regionList;

ProtRegion::ProtRegion(MemHandler &memHandler, void *addr, size_t size) :
	MemRegion(addr, size),
	memHandler(memHandler),
	access(0),
	dirty(false),
	permission(ReadWrite)
{
	if(regionList.empty()) setHandler();
	regionList.push_front(this);
	TRACE("New ProtRegion %p (%d bytes)", addr, size);
}

ProtRegion::~ProtRegion()
{
	std::list<ProtRegion *>::iterator i;
	i = std::find(regionList.begin(), regionList.end(), this);
	if(i != regionList.end()) regionList.erase(i);
	if(regionList.empty()) restoreHandler();
}

};
