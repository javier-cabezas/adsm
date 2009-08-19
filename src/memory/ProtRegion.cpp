#include "ProtRegion.h"

#include <debug.h>

namespace gmac {
unsigned ProtRegion::count = 0;

ProtRegion::ProtRegion(void *addr, size_t size) :
	MemRegion(addr, size),
	_dirty(false),
	_present(true)
{
	if(count == 0) setHandler();
	count++;
	TRACE("New ProtRegion %p (%d bytes)", addr, size);
}

ProtRegion::~ProtRegion()
{
	count--;
	if(count == 0) restoreHandler();
}

};
