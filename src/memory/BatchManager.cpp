#include "BatchManager.h"

#include <debug.h>

#include <kernel/Context.h>

namespace gmac {
void BatchManager::release(void *addr)
{
	MemRegion *reg = remove(addr);
	unmap(reg->start(), reg->size());
	delete reg;
}
void BatchManager::flush(void)
{
	MemMap::const_iterator i;
	current().lock();
	for(i = current().begin(); i != current().end(); i++) {
		TRACE("Memory Copy to Device");
		Context::current()->copyToDevice(safe(i->second->start()),
				i->second->start(), i->second->size());
	}
	current().unlock();
}

void BatchManager::sync(void)
{
	MemMap::const_iterator i;
	current().lock();
	for(i = current().begin(); i != current().end(); i++) {
		TRACE("Memory Copy from Device");
		Context::current()->copyToHost(i->second->start(),
				safe(i->second->start()), i->second->size());
	}
	current().unlock();
}

};
