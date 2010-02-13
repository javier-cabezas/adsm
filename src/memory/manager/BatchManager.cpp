#include "BatchManager.h"

#include <debug.h>

#include <kernel/Context.h>

namespace gmac { namespace memory { namespace manager {

void BatchManager::release(void *addr)
{
	Region *reg = remove(ptr(addr));
	hostUnmap(reg->start(), reg->size());
	removeVirtual(reg->start(), reg->size());
	delete reg;
}
void BatchManager::flush(void)
{
	Map::const_iterator i;
	current()->lock();
    Context * ctx = Context::current();
	for(i = current()->begin(); i != current()->end(); i++) {
		TRACE("Memory Copy to Device");
		ctx->copyToDevice(ptr(i->second->start()),
				i->second->start(), i->second->size());
	}
	current()->unlock();
	ctx->flush();
	ctx->sync();
}

void BatchManager::sync(void)
{
	Map::const_iterator i;
	current()->lock();
	for(i = current()->begin(); i != current()->end(); i++) {
		TRACE("Memory Copy from Device");
		Context::current()->copyToHost(i->second->start(),
				ptr(i->second->start()), i->second->size());
	}
	current()->unlock();
}

void
BatchManager::invalidate(const void *, size_t)
{
    // Do nothing
}

void
BatchManager::flush(const void *, size_t)
{
    // Do nothing
}

}}}
