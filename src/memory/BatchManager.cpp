#include "BatchManager.h"
#include "paraver.h"

#include <config/debug.h>
#include <acc/api.h>

namespace gmac {
void BatchManager::release(void *addr)
{
	MemRegion *reg = memMap.remove(addr);
	unmap(reg->getAddress(), reg->getSize());
	delete reg;
}
void BatchManager::flush(void)
{
	MemMap<MemRegion>::const_iterator i;
	memMap.lock();
	for(i = memMap.begin(); i != memMap.end(); i++) {
		if(i->second->isOwner() == false) continue;	
		TRACE("Memory Copy to Device");
		__gmacMemcpyToDevice(safe(i->second->getAddress()),
				i->second->getAddress(), i->second->getSize());
	}
	memMap.unlock();
}

void BatchManager::sync(void)
{
	MemMap<MemRegion>::const_iterator i;
	memMap.lock();
	for(i = memMap.begin(); i != memMap.end(); i++) {
		if(i->second->isOwner() == false) continue;	
		TRACE("Memory Copy from Device");
		__gmacMemcpyToHost(i->second->getAddress(),
				safe(i->second->getAddress()), i->second->getSize());
	}
	memMap.unlock();
}

};
