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

    gmac::Context *srcCtx = manager->owner(buf);
    gmacError_t err;

	pushState(IORead);
    size_t ret = 0;
    if(srcCtx == NULL) {
        ret = __libc_fread(buf, size, nmemb, stream);
    } else {
        manager->invalidate(buf, n);
        void * tmp;

        gmac::Context *ctx = gmac::Context::current();
        if (ctx->bufferPageLockedSize() > 0) {
            size_t bufferSize = ctx->bufferPageLockedSize();
            tmp = ctx->bufferPageLocked();

            size_t left = n;
            off_t  off  = 0;
            while (left != 0) {
                size_t bytes= left < bufferSize? left: bufferSize;

                TRACE("Reading: %zd\n", bytes);
                ret += __libc_fread(tmp, size, bytes/size, stream);
                err = srcCtx->copyToDevice(manager->ptr(((char *) buf) + off), tmp, bytes);
                assert(err == gmacSuccess);

                left -= bytes;
                off  += bytes;
            }
        } else {
            tmp = malloc(n);
            ret = __libc_fread(tmp, size, nmemb, stream);
            err = srcCtx->copyToDevice(manager->ptr(buf), tmp, n);
            assert(err == gmacSuccess);
            free(tmp);
        }
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

    gmac::Context *dstCtx = manager->owner(buf);
    gmacError_t err;

    pushState(IOWrite);
    size_t ret = 0;
    if(dstCtx == NULL) {
        ret = __libc_fwrite(buf, size, nmemb, stream);
    } else {
        manager->flush(buf, n);
        void * tmp;

        gmac::Context *ctx = gmac::Context::current();
        if (ctx->bufferPageLockedSize() > 0) {
            size_t bufferSize = ctx->bufferPageLockedSize();
            tmp               = ctx->bufferPageLocked();
            printf("IO Address %p\n", tmp);

            size_t left = n;
            off_t  off  = 0;
            while (left != 0) {
                size_t bytes = left < bufferSize? left: bufferSize;

                TRACE("Writing: %zd\n", bytes);
                err = dstCtx->copyToHost(tmp, manager->ptr(((char *) buf) + off), bytes);
                assert(err == gmacSuccess);
                ret += __libc_fwrite(tmp, size, bytes/size, stream);

                left -= bytes;
                off  += bytes;
            }
        } else {
            tmp = malloc(n);
            err = dstCtx->copyToHost(tmp, manager->ptr(buf), n);
            assert(err == gmacSuccess);
            ret =  __libc_fwrite(tmp, size, nmemb, stream);
            free(tmp);
        }
    }
    popState();

    return ret;
}
