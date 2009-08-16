#include <os/loader.h>

#include <paraver.h>
#include <debug.h>

#include <unistd.h>
#include <stdio.h>
#include <stdint.h>
#include <dlfcn.h>

#include <errno.h>

SYM(ssize_t, __libc_read, int, void *, size_t);
SYM(ssize_t, __libc_write, int, const void *, size_t);

static void __attribute__((constructor(101))) posixIoInit(void)
{
	LOAD_SYM(__libc_read, read);
	LOAD_SYM(__libc_write, write);
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

#ifdef __cplusplus
}
#endif
