#include <os/loader.h>

#include <order.h>
#include <paraver.h>
#include <debug.h>

#include <init.h>
#include <memory/Manager.h>
#include <kernel/Context.h>

#include <unistd.h>
#include <stdio.h>
#include <stdint.h>
#include <dlfcn.h>

#include <errno.h>

SYM(ssize_t, __libc_read, int, void *, size_t);
SYM(ssize_t, __libc_write, int, const void *, size_t);

static void __attribute__((constructor(INTERPOSE))) posixIoInit(void)
{
	TRACE("Overloading I/O POSIX functions");
	LOAD_SYM(__libc_read, read);
	LOAD_SYM(__libc_write, write);
}



/* System call wrappers */

#ifdef __cplusplus
extern "C"
#endif
ssize_t read(int fd, void *buf, size_t count)
{
	if(__libc_read == NULL) posixIoInit();
	if(__inGmac() == 1 || manager == NULL) return __libc_read(fd, buf, count);

   gmac::Context *ctx = manager->owner(buf);

   if(ctx == NULL) return __libc_read(fd, buf, count);

   __enterGmac();
	pushState(IORead);

    manager->invalidate(buf, count);
    void *tmp = malloc(count);

    size_t ret = __libc_read(fd, tmp, count);
    ctx->copyToDevice(manager->ptr(buf), tmp, count);
    free(tmp);

    popState();
	__exitGmac();

    return ret;
}

#ifdef __cplusplus
extern "C"
#endif
ssize_t write(int fd, const void *buf, size_t count)
{
	if(__libc_read == NULL) posixIoInit();
	if(__inGmac() == 1 || manager == NULL) return __libc_write(fd, buf, count);

    gmac::Context *ctx = manager->owner(buf);

    if(ctx == NULL) return __libc_write(fd, buf, count);

	__enterGmac();
    pushState(IOWrite);

    void *tmp = malloc(count);

    manager->flush(buf, count);
    ctx->copyToHost(tmp, manager->ptr(buf), count);

    size_t ret =  __libc_write(fd, tmp, count);
    free(tmp);
	
    popState();
	__exitGmac();

    return ret;
}
