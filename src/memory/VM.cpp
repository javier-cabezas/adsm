#include "VM.h"

#include <kernel/Context.h>

namespace gmac { namespace memory  { namespace vm {

void *Dumper::alloc(size_t size) const
{
	void *device = NULL;
#ifdef USE_VM
	if(gmac::Context::current() == NULL) return NULL;
	assert(gmac::Context::current()->malloc((void **)&device,
		size) == gmacSuccess);
#endif
	return device;
}

void *Dumper::host_alloc(void **host, size_t size) const
{
	void *device = NULL;
	*host = NULL;
#ifdef USE_VM
	if(gmac::Context::current() == NULL) return NULL;
	assert(gmac::Context::current()->hostost__alloc(host, (void **)&device,
		size) == gmacSuccess);
	return device;
#endif
}

void Dumper::free(void *addr) const
{
#ifdef USE_VM
	if(gmac::Context::current() == NULL) return;
	assert(gmac::Context::current()->free(addr) == gmacSuccess);
#endif
}

void Dumper::host_free(void *addr) const
{
#ifdef USE_VM
	if(gmac::Context::current() == NULL) return;
	assert(gmac::Context::current()->host_free(addr) == gmacSuccess);
#endif
}

void Dumper::flush(void *dst, const void *src, size_t s) const
{
#ifdef USE_VM
	assert(gmac::Context::current()->copyToDevice(dst, src, s) == gmacSuccess);
#endif
}

void Dumper::sync(void *dst, const void *src, size_t s) const
{
#ifdef USE_VM
	assert(gmac::Context::current()->copyToHost(dst, src, s) == gmacSuccess);
#endif
}

}}};
