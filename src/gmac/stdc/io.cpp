#include <os/loader.h>

#include <order.h>
#include <debug.h>
#include <params.h>
#include <paraver.h>

#include <init.h>
#include <memory/Manager.h>
#include <kernel/Context.h>

#include <unistd.h>
#include <stdio.h>
#include <stdint.h>
#include <dlfcn.h>

#include <errno.h>

SYM(size_t, __libc_fread, void *, size_t, size_t, FILE *);
SYM(size_t, __libc_fwrite, const void *, size_t, size_t, FILE *);

static void __attribute__((constructor(INTERPOSE))) stdcInit(void)
{
	TRACE("Overloading I/O STDC functions");
	LOAD_SYM(__libc_fread, fread);
	LOAD_SYM(__libc_fwrite, fwrite);
}

/* Standard C library wrappers */

#ifdef __cplusplus
extern "C"
#endif
size_t fread(void *buf, size_t size, size_t nmemb, FILE *stream)
{
    size_t n = size * nmemb;

    gmac::Context *ctx = manager->owner(buf);
    gmacError_t err;

    if(ctx == NULL) return __libc_fread(buf, size, nmemb, stream);

	pushState(IORead);

    manager->invalidate(buf, n);
    void * tmp;
    size_t ret;

    if (ctx->bufferPageLockedSize() > 0) {
        size_t bufferSize = ctx->bufferPageLockedSize();
        tmp = ctx->bufferPageLocked();

        size_t left = n;
        off_t  off  = 0;
        while (left != 0) {
            size_t size = left < bufferSize? left: bufferSize;

            TRACE("Reading: %zd\n", size);
            ret = __libc_fread(tmp, size, 1, stream);
            err = ctx->copyToDevice(manager->ptr(((char *) buf) + off), tmp, size);
            assert(err == gmacSuccess);

            left -= size;
            off  += size;
        }
    } else {
        tmp = malloc(n);
        ret = __libc_fread(tmp, size, nmemb, stream);
        err = ctx->copyToDevice(manager->ptr(buf), tmp, n);
        assert(err == gmacSuccess);
        free(tmp);
    }

    popState();

    return ret;
}


#ifdef __cplusplus
extern "C"
#endif
size_t fwrite(const void *buf, size_t size, size_t nmemb, FILE *stream)
{
    size_t n = size * nmemb;

    gmac::Context *ctx = manager->owner(buf);
    gmacError_t err;

    if(ctx == NULL) return __libc_fwrite(buf, size, nmemb, stream);

    pushState(IOWrite);

    manager->flush(buf, n);
    void * tmp;
    size_t ret;

    if (ctx->bufferPageLockedSize() > 0) {
        size_t bufferSize = ctx->bufferPageLockedSize();
        tmp = ctx->bufferPageLocked();

        size_t left = n;
        off_t  off  = 0;
        while (left != 0) {
            size_t size = left < bufferSize? left: bufferSize;

            TRACE("Writing: %zd\n", size);
            err = ctx->copyToHost(tmp, manager->ptr(((char *) buf) + off), size);
            assert(err == gmacSuccess);
            ret = __libc_fwrite(tmp, size, 1, stream);

            left -= size;
            off  += size;
        }
    } else {
        tmp = malloc(n);
        err = ctx->copyToHost(tmp, manager->ptr(buf), n);
        assert(err == gmacSuccess);
        ret =  __libc_fwrite(tmp, size, nmemb, stream);
        free(tmp);
    }
	
    popState();

    return ret;
}
