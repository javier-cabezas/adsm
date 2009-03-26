#include <gmac.h>
#include <os/loader.h>

#include <string.h>

#include <common/MemManager.h>

SYM(void *, __libc_memset, void *, int, size_t);
SYM(void *, __libc_memcpy, void *, const void *, size_t);

extern gmac::MemManager *memManager;

static void __attribute__((constructor)) stdcMemInit(void)
{
	LOAD_SYM(__libc_memset, memset);
	LOAD_SYM(__libc_memcpy, memcpy);
}

void *memset(void *s, int c, size_t n)
{
	return __libc_memset(s, c, n);
}

void *memcpy(void *dst, const void *src, size_t n)
{
	return __libc_memcpy(dst, src, n);
}
