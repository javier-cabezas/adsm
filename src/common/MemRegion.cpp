#include "MemRegion.h"
#include "debug.h"

#include <string.h>

#include <algorithm>

namespace gmac {
MemHandler *MemHandler::handler = NULL;
unsigned ProtRegion::count = 0;

ProtRegion::ProtRegion(void *addr, size_t size) :
	MemRegion(addr, size),
	dirty(false),
	present(true)
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

CacheRegion::CacheRegion(void *addr, size_t size, size_t cacheLine) :
	ProtRegion(addr, size),
	cacheLine(cacheLine),
	offset((unsigned long)addr & (cacheLine -1))
{
	for(size_t s = 0; s < size; s += cacheLine) {
		void *p = (void *)((uint8_t *)addr + s);
		size_t regionSize = ((s - size) > cacheLine) ? cacheLine : (s - size);
		set[p] = new ProtSubRegion(this, p, regionSize);
	}
	present = set;
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
	Set::const_iterator i;
	for(i = present.begin(); i != present.end();) {
		Set::const_iterator current = i;
		i++;
		current->second->noAccess();
	}
}


};
