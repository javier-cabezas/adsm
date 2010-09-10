#include <os/loader.h>

#include <paraver.h>

#include <init.h>
#include <memory/Manager.h>
#include <kernel/IOBuffer.h>
#include <kernel/Mode.h>
#include <trace/Thread.h>

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

    __enterGmac();
    gmac::Mode *dstMode = proc->owner(buf);

    if(dstMode == NULL) {
        __exitGmac();
        return  __libc_fread(buf, size, nmemb, stream);
    }
	
    gmac::trace::Thread::io();
    gmacError_t err;
    size_t ret = 0;

    manager->invalidate(buf, n);

    gmac::IOBuffer *buffer = proc->createIOBuffer(paramPageSize);
    
    size_t left = n;
    off_t  off  = 0;
    while (left != 0) {
        size_t bytes= left < buffer->size()? left: buffer->size();
        ret += __libc_fread(buffer->addr(), size, bytes/size, stream);
        err = dstMode->bufferToDevice(buffer, proc->translate((char *)buf + off), bytes);
        gmac::util::Logger::ASSERTION(err == gmacSuccess);

        left -= bytes;
        off  += bytes;
    }
    proc->destroyIOBuffer(buffer);
    gmac::trace::Thread::resume();
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

	__enterGmac();

    gmac::trace::Thread::io();
    gmacError_t err;
    size_t n = size * nmemb;
    size_t ret = 0;

    manager->release((void *)buf, n);

    off_t  off  = 0;
    gmac::IOBuffer *buffer = proc->createIOBuffer(paramPageSize);

    size_t left = n;
    buffer->lock();
    while (left != 0) {
        size_t bytes = left < buffer->size() ? left : buffer->size();
        err = srcMode->bufferToHost(buffer, proc->translate((char *)buf + off), bytes);
        gmac::util::Logger::ASSERTION(err == gmacSuccess);
        ret += __libc_fwrite(buffer->addr(), size, bytes/size, stream);

        left -= bytes;
        off  += bytes;
    }
    proc->destroyIOBuffer(buffer);
    gmac::trace::Thread::resume();
	__exitGmac();

    return ret;
}
