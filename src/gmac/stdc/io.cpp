#include <os/loader.h>

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

    gmac::Mode *dstMode = proc->owner(buf);

    if(dstMode == NULL) return  __libc_fread(buf, size, nmemb, stream);
    gmac::Context &dstCtx = dstMode->context();
	
    __enterGmac();
	pushState(IORead);

    gmacError_t err;
    size_t ret = 0;

    manager->invalidate(buf, n);

    gmac::Mode *mode = gmac::Mode::current();
    gmac::Context &ctx = mode->context();

    size_t bufferSize = ctx.bufferPageLockedSize();
    void * tmp = ctx.bufferPageLocked();
    ctx.lockRead();
    if(&ctx != &dstCtx) dstCtx.lockRead();
    size_t left = n;
    off_t  off  = 0;
    while (left != 0) {
        size_t bytes= left < bufferSize? left: bufferSize;

        ret += __libc_fread(tmp, size, bytes/size, stream);
        err = dstCtx.copyToDeviceAsync(dstMode->translate(((char *) buf) + off), tmp, bytes);
        gmac::util::Logger::ASSERTION(err == gmacSuccess);
        err = dstCtx.syncToDevice();
        gmac::util::Logger::ASSERTION(err == gmacSuccess);

        left -= bytes;
        off  += bytes;
    }
    if(&ctx != &dstCtx) dstCtx.unlock();
    ctx.unlock();
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

    gmac::Mode *srcMode = proc->owner(buf);

    if(srcMode == NULL) return __libc_fwrite(buf, size, nmemb, stream);
    gmac::Context &srcCtx = srcMode->context();

	__enterGmac();
    pushState(IOWrite);

    gmacError_t err;
    size_t n = size * nmemb;
    size_t ret = 0;

    manager->release((void *)buf, n);

    gmac::Mode *mode = gmac::Mode::current();
    gmac::Context &ctx = mode->context();
    ctx.lockRead();
    off_t  off  = 0;
    size_t bufferSize = ctx.bufferPageLockedSize();
    void * tmp        = ctx.bufferPageLocked();
    size_t left = n;
    if(&srcCtx != &ctx) srcCtx.lockRead();
    while (left != 0) {
        size_t bytes = left < bufferSize ? left : bufferSize;
        err = srcCtx.copyToHostAsync(tmp, srcMode->translate(((char *) buf) + off), bytes);
        gmac::util::Logger::ASSERTION(err == gmacSuccess);
        err = srcCtx.syncToHost();
        gmac::util::Logger::ASSERTION(err == gmacSuccess);
        ret += __libc_fwrite(tmp, size, bytes/size, stream);

        left -= bytes;
        off  += bytes;
    }
    if(&srcCtx != &ctx) srcCtx.unlock();
    ctx.unlock();
    popState();
	__exitGmac();

    return ret;
}
