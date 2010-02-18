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
	assert(ctx->malloc((void **)&device,
		size) == gmacSuccess);
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
	assert(ctx->hostAlloc(host, (void **)&device,
		size) == gmacSuccess);
	return device;
#endif
}

void
Dumper::free(void *addr) const
{
#ifdef USE_VM
    Context * ctx == Context::current();
	if(ctx == NULL) return;
	assert(ctx->free(addr) == gmacSuccess);
#endif
}

void
Dumper::hostFree(void *addr) const
{
#ifdef USE_VM
    Context * ctx = Context::current();
	if(ctx == NULL) return;
	assert(ctx->hostFree(addr) == gmacSuccess);
#endif
}

void
Dumper::flush(void *dst, const void *src, size_t s) const
{
#ifdef USE_VM
	assert(Context::current()->copyToDevice(dst, src, s) == gmacSuccess);
#endif
}

void
Dumper::sync(void *dst, const void *src, size_t s) const
{
#ifdef USE_VM
	assert(Context::current()->copyToHost(dst, src, s) == gmacSuccess);
#endif
}

}}};
