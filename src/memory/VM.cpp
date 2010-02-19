#include "VM.h"

#include <kernel/Context.h>

namespace gmac { namespace memory  { namespace vm {

void *
Dumper::alloc(size_t size) const
{
	void *device = NULL;
#ifdef USE_VM
    Context * ctx = Context::current();
	if(ctx == NULL) return NULL;
    gmacError_t ret = ctx->malloc((void **)&device, size);
	ASSERT(ret == gmacSuccess);
#endif
	return device;
}

void *
Dumper::hostAlloc(void **host, size_t size) const
{
	void *device = NULL;
	*host = NULL;
#ifdef USE_VM
    Context * ctx = Context::current();
	if(ctx == NULL) return NULL;
    gmacError_t ret = ctx->hostAlloc(host, (void **)&device, size);
	ASSERT(ret == gmacSuccess);
	return device;
#endif
}

void
Dumper::free(void *addr) const
{
#ifdef USE_VM
    Context * ctx == Context::current();
	if(ctx == NULL) return;
    gmacError_t ret = ctx->free(addr);
	ASSERT(ret == gmacSuccess);
#endif
}

void
Dumper::hostFree(void *addr) const
{
#ifdef USE_VM
    Context * ctx = Context::current();
	if(ctx == NULL) return;
    gmacError_t ret = ctx->hostFree(addr);
	ASSERT(ret == gmacSuccess);
#endif
}

void
Dumper::flush(void *dst, const void *src, size_t s) const
{
#ifdef USE_VM
    gmacError_t ret = Context::current()->copyToDevice(dst, src, s);
	ASSERT(ret == gmacSuccess);
#endif
}

void
Dumper::sync(void *dst, const void *src, size_t s) const
{
#ifdef USE_VM
    gmacError_t ret = Context::current()->copyToHost(dst, src, s);
	ASSERT(ret == gmacSuccess);
#endif
}

}}}
