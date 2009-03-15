#include <loader.h>
#include <common/debug.h>
#include <common/paraver.h>

#include <unistd.h>
#include <stdio.h>
#include <stdint.h>
#include <dlfcn.h>

#include <errno.h>

SYM(ssize_t, __libc_read, int, void *, size_t);
SYM(ssize_t, __libc_write, int, const void *, size_t);
SYM(size_t, __libc_fread, void *, size_t, size_t, FILE *);
SYM(size_t, __libc_fwrite, const void *, size_t, size_t, FILE *);

static void __attribute__((constructor)) ioInit(void)
{
	LOAD_SYM(__libc_read, read);
	LOAD_SYM(__libc_write, write);
	LOAD_SYM(__libc_fread, fread);
	LOAD_SYM(__libc_fwrite, fwrite);
}


#ifdef __cplusplus
extern "C" {
#endif

/* System call wrappers */

ssize_t read(int fd, void *buf, size_t count)
{
	pushState(_IORead_);
	ssize_t n = 0;
	uint8_t *ptr = (uint8_t *)buf;
	do {
		n += __libc_read(fd, ptr + n, count - n);
	} while(n < count && errno == EFAULT);
	popState();
	return n;
}


ssize_t write(int fd, const void *buf, size_t count)
{
	pushState(_IOWrite_);
	ssize_t n = 0;
	uint8_t *ptr = (uint8_t *)buf;
	do {
		n += __libc_write(fd, ptr + n, count - n);
	} while(n < count && errno == EFAULT);
	popState();
	return n;
}

/* Standard C library wrappers */

size_t fread(void *buf, size_t size, size_t nmemb, FILE *stream)
{
	pushState(_IORead_);
	ssize_t n = 0;
	uint8_t *ptr = (uint8_t *)buf;
	do {
		n += __libc_fread(ptr + (n * size), size, nmemb - n, stream);
		if(ferror(stream) && errno == EFAULT) clearerr(stream);
	} while(n < nmemb && ferror(stream) == 0);
	popState();
	return n;
}

#include <string.h>

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
