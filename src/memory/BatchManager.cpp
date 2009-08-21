#include "BatchManager.h"

#include <debug.h>

#include <kernel/Context.h>

namespace gmac {
void BatchManager::release(void *addr)
{
	MemRegion *reg = remove(addr);
	unmap(reg->getAddress(), reg->getSize());
	delete reg;
}
void BatchManager::flush(void)
{
	MemMap::const_iterator i;
	MemMap &mm = Context::current()->mm();
	mm.lock();
	for(i = mm.begin(); i != mm.end(); i++) {
		TRACE("Memory Copy to Device");
		Context::current()->copyToDevice(safe(i->second->getAddress()),
				i->second->getAddress(), i->second->getSize());
	}
	mm.unlock();
}

void BatchManager::sync(void)
{
	MemMap::const_iterator i;
	MemMap &mm = Context::current()->mm();
	mm.lock();
	for(i = mm.begin(); i != mm.end(); i++) {
		TRACE("Memory Copy from Device");
		Context::current()->copyToHost(i->second->getAddress(),
				safe(i->second->getAddress()), i->second->getSize());
	}
	mm.unlock();
}

};
