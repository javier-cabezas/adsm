#include <os/loader.h>

#include <config/debug.h>
#include <config/paraver.h>

#include <unistd.h>
#include <stdio.h>
#include <stdint.h>
#include <dlfcn.h>

#include <errno.h>

SYM(size_t, __libc_fread, void *, size_t, size_t, FILE *);
SYM(size_t, __libc_fwrite, const void *, size_t, size_t, FILE *);

static void __attribute__((constructor(101))) stdcInit(void)
{
	LOAD_SYM(__libc_fread, fread);
	LOAD_SYM(__libc_fwrite, fwrite);
}


#ifdef __cplusplus
extern "C" {
#endif

/* Standard C library wrappers */

size_t fread(void *buf, size_t size, size_t nmemb, FILE *stream)
{
	pushState(_IORead_);
	ssize_t n = 0;
	uint8_t *ptr = (uint8_t *)buf;
	do {
		n += __libc_fread(ptr + (n * size), size, nmemb - n, stream);
		if(feof(stream)) break;
		if(ferror(stream) && errno == EFAULT) clearerr(stream);
	} while(n < nmemb && ferror(stream) == 0);
	popState();
	return n;
}


size_t fwrite(const void *buf, size_t size, size_t nmemb, FILE *stream)
{
	pushState(_IOWrite_);
	ssize_t n = 0;
	uint8_t *ptr = (uint8_t *)buf;
	do {
		n += __libc_fwrite(ptr + (n * size), size, nmemb - n, stream);
		if(ferror(stream) && errno == EFAULT) clearerr(stream);
	} while(n < nmemb && ferror(stream) == 0);
	popState();
	return n;
}


#ifdef __cplusplus
}
#endif
