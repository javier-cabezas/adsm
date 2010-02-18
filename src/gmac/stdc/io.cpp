#include <os/loader.h>

#include <debug.h>
#include <paraver.h>

#include <init.h>
#include <memory/Manager.h>
#include <kernel/Context.h>

#include <unistd.h>
#include <cstdio>
#include <stdint.h>
#include <dlfcn.h>

#include <errno.h>

#include "stdc.h"

SYM(size_t, __libc_fread, void *, size_t, size_t, FILE *);
SYM(size_t, __libc_fwrite, const void *, size_t, size_t, FILE *);

void stdcIoInit(void)
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
	if(__libc_fread == NULL) stdcIoInit();
	if(__inGmac() == 1) return __libc_fread(buf, size, nmemb, stream);
    size_t n = size * nmemb;

    gmac::Context *srcCtx = manager->owner(buf);

    if(srcCtx == NULL) return  __libc_fread(buf, size, nmemb, stream);
	
    __enterGmac();
	pushState(IORead);
    if (srcCtx->status() == gmac::Context::RUNNING) srcCtx->sync();

    gmacError_t err;
    size_t ret = 0;

    manager->invalidate(buf, n);

    gmac::Context *ctx = gmac::Context::current();

    if (ctx->async()) {
        size_t bufferSize = ctx->bufferPageLockedSize();
        void * tmp = ctx->bufferPageLocked();

        size_t left = n;
        off_t  off  = 0;
        while (left != 0) {
            size_t bytes= left < bufferSize? left: bufferSize;

            ret += __libc_fread(tmp, size, bytes/size, stream);
            err = srcCtx->copyToDevice(manager->ptr(((char *) buf) + off), tmp, bytes);
            assert(err == gmacSuccess);

            left -= bytes;
            off  += bytes;
        }
    } else {
        void * tmp = malloc(n);
        ret = __libc_fread(tmp, size, nmemb, stream);
        err = srcCtx->copyToDevice(manager->ptr(buf), tmp, n);
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
size_t fwrite(const void *buf, size_t size, size_t nmemb, FILE *stream)
{
	if(__libc_fwrite == NULL) stdcIoInit();
	if(__inGmac() == 1) return __libc_fwrite(buf, size, nmemb, stream);

    gmac::Context *dstCtx = manager->owner(buf);

    if(dstCtx == NULL) return __libc_fwrite(buf, size, nmemb, stream);

	__enterGmac();
    pushState(IOWrite);

    gmacError_t err;
    size_t n = size * nmemb;
    size_t ret = 0;

    //manager->flush(buf, n);

    gmac::Context *ctx = gmac::Context::current();
    if (ctx->async()) {
        size_t bufferSize = ctx->bufferPageLockedSize();
        void * tmp        = ctx->bufferPageLocked();

        size_t left = n;
        off_t  off  = 0;
        while (left != 0) {
            size_t bytes = left < bufferSize? left: bufferSize;

            err = dstCtx->copyToHost(tmp, manager->ptr(((char *) buf) + off), bytes);
            assert(err == gmacSuccess);
            ret += __libc_fwrite(tmp, size, bytes/size, stream);

            left -= bytes;
            off  += bytes;
        }
    } else {
        void * tmp = malloc(n);
        err = dstCtx->copyToHost(tmp, manager->ptr(buf), n);
        assert(err == gmacSuccess);
        ret =  __libc_fwrite(tmp, size, nmemb, stream);
        free(tmp);
    }
    popState();
	__exitGmac();

    return ret;
}
