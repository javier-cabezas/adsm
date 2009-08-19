#include <os/loader.h>
#include <order.h>
#include <gmac.h>

#include <string.h>

#include <memory/MemManager.h>
#include <kernel/Context.h>


SYM(void *, __libc_memset, void *, int, size_t);
SYM(void *, __libc_memcpy, void *, const void *, size_t);

extern gmac::MemManager *memManager;

static void __attribute__((constructor(INTERPOSE))) stdcMemInit(void)
{
	TRACE("Overloading Memory STDC functions");
	LOAD_SYM(__libc_memset, memset);
	LOAD_SYM(__libc_memcpy, memcpy);
}


void *memset(void *s, int c, size_t n)
{
	gmac::MemRegion *r = NULL;

	if(memManager == NULL) return __libc_memset(s, c, n);
	while(n) {
		size_t size = memManager->filter(s, n, r);
		if(r) {
			if(size != r->getSize()) memManager->flush(r);
			r->context()->memset(memManager->safe(s), c, size);
			memManager->invalidate(r);
			TRACE("memset %p [device]", r->getAddress());
		}
		else __libc_memset(s, c, size);
		// Update size and address
		n -= size;
		s = (void *)((unsigned long)s + size);
	}
}

void *memcpy(void *dst, const void *src, size_t n)
{
	void *ret = dst;
	gmac::MemRegion *d = NULL, *s = NULL;
	size_t ds = 0, ss = 0;

	if(memManager == NULL) return __libc_memcpy(dst, src, n);
	while(n) {
		if(ds == 0) ds = memManager->filter(dst, n, d);
		if(ss == 0) ss = memManager->filter(src, n, s);
		size_t size = (ds > ss) ? ss : ds;
		// If both memories are shared and both of them are
		// in device memory, use a internal copy
		if(d != NULL && memManager->present(d) == true &&
			s != NULL && memManager->present(s) == true) {
			TRACE("memcpy %p to %p [DeviceToDevice]", src, dst);
			assert(d->context() == s->context());
			d->context()->copyDevice(memManager->safe(dst),
				memManager->safe(src), size);
		}
		// If the destination is shared memory and it is on the
		// device, copy it to device memory
		else if(d != NULL && memManager->present(d) == false) {
			TRACE("memcpy %p to %p [HostToDevice]", src, dst);
			d->context()->copyToDeviceAsync(memManager->safe(dst),
					src, size);
		}
		// If the source is shared memory and it is on the device,
		// copy it from device memory
		else if(s != NULL && memManager->present(s) == false) {
			TRACE("memcpy %p to %p [DeviceToHost]", src, dst);
			s->context()->copyToHost(dst, memManager->safe(src),
					size);
		}
		// Both (src and dst) are not shared or they are present
		// in system memory
		else {
			__libc_memcpy(dst, src, size);
		}
		// Update sizes and addresses
		n -= size; ds -= size; ss -= size;
		dst = (void *)((unsigned long)dst + size);
		src = (void *)((unsigned long)src + size);
	}

	return ret;
}
