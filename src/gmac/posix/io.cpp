#include <os/loader.h>

#include <order.h>
#include <paraver.h>
#include <debug.h>

#include <init.h>
#include <memory/Manager.h>
#include <kernel/Context.h>

#include <unistd.h>
#include <cstdio>
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

   gmac::Context *srcCtx = manager->owner(buf);

   if(srcCtx == NULL) return __libc_read(fd, buf, count);

   __enterGmac();
	pushState(IORead);

    gmacError_t err;
    size_t ret = 0;

    manager->invalidate(buf, count);

    gmac::Context *ctx = gmac::Context::current();
    if (ctx->async()) {
        size_t bufferSize = ctx->bufferPageLockedSize();
        void * tmp = ctx->bufferPageLocked();

        size_t left = count;
        off_t  off  = 0;
        while (left != 0) {
            size_t bytes= left < bufferSize? left: bufferSize;

            ret += __libc_read(fd, tmp, bytes);
            err = srcCtx->copyToDevice(manager->ptr(((char *) buf) + off), tmp, bytes);
            assert(err == gmacSuccess);

            left -= bytes;
            off  += bytes;
        }
    } else {
        void * tmp = malloc(count);
        ret = __libc_read(fd, tmp, count);
        err = srcCtx->copyToDevice(manager->ptr(buf), tmp, count);
        assert(err == gmacSuccess);
        free(tmp);
    }

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

    gmac::Context *dstCtx = manager->owner(buf);

    if(dstCtx == NULL) return __libc_write(fd, buf, count);

	__enterGmac();
    pushState(IOWrite);

    gmacError_t err;
    size_t ret = 0;

    manager->flush(buf, count);

    gmac::Context *ctx = gmac::Context::current();
    if (ctx->async()) {
        size_t bufferSize = ctx->bufferPageLockedSize();
        void * tmp        = ctx->bufferPageLocked();

        size_t left = count;
        off_t  off  = 0;
        while (left != 0) {
            size_t bytes = left < bufferSize? left: bufferSize;

            err = dstCtx->copyToHost(tmp, manager->ptr(((char *) buf) + off), bytes);
            assert(err == gmacSuccess);
            ret += __libc_write(fd, tmp, bytes);

            left -= bytes;
            off  += bytes;
        }
    } else {
        void *tmp = malloc(count);
        err = dstCtx->copyToHost(tmp, manager->ptr(buf), count);
        assert(err == gmacSuccess);
        ret =  __libc_write(fd, tmp, count);
        free(tmp);
    }
	
    popState();
	__exitGmac();

    return ret;
}
