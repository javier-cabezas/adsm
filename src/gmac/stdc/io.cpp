#if defined(POSIX)
#include "os/posix/loader.h"
#elif defined(WINDOWS)
#include "os/windows/loader.h"
#endif

#include "gmac/paraver.h"

#include "gmac/init.h"
#include "memory/Manager.h"
#include "core/IOBuffer.h"
#include "core/Process.h"
#include "core/Mode.h"
#include "trace/Thread.h"
#include "util/Logger.h"

#include <cstdio>

#include <errno.h>

#include "stdc.h"

SYM(size_t, __libc_fread, void *, size_t, size_t, FILE *);
SYM(size_t, __libc_fwrite, const void *, size_t, size_t, FILE *);


#ifdef __cplusplus
extern "C"
#endif
size_t SYMBOL(fread)(void *buf, size_t size, size_t nmemb, FILE *stream)
{
	if(__libc_fread == NULL) stdcIoInit();
	if(gmac::inGmac() == 1) return __libc_fread(buf, size, nmemb, stream);
    size_t n = size * nmemb;

    gmac::enterGmac();
    gmac::Process &proc = gmac::Process::getInstance();
    gmac::Mode *dstMode = proc.owner(buf);

    if(dstMode == NULL) {
        gmac::exitGmac();
        return  __libc_fread(buf, size, nmemb, stream);
    }
	
    gmac::trace::Thread::io();
    gmacError_t err;
    size_t ret = 0;
    gmac::IOBuffer *buffer = proc.createIOBuffer(paramPageSize);
    gmac::memory::Manager &manager = gmac::memory::Manager::getInstance();
    
    size_t left = n;
    off_t  off  = 0;
    while (left != 0) {
        size_t bytes= left < buffer->size()? left: buffer->size();
        ret += __libc_fread(buffer->addr(), size, bytes/size, stream);
        err = manager.fromIOBuffer((char *)buf + off, *buffer,  bytes);
        gmac::util::Logger::ASSERTION(err == gmacSuccess);
        err = buffer->wait();
        gmac::util::Logger::ASSERTION(err == gmacSuccess);

        left -= bytes;
        off  += (off_t)bytes;
    }
    proc.destroyIOBuffer(buffer);
    gmac::trace::Thread::resume();
	gmac::exitGmac();

    return ret;
}


#ifdef __cplusplus
extern "C"
#endif
size_t SYMBOL(fwrite)(const void *buf, size_t size, size_t nmemb, FILE *stream)
{
	if(__libc_fwrite == NULL) stdcIoInit();
	if(gmac::inGmac() == 1) return __libc_fwrite(buf, size, nmemb, stream);

    gmac::Process &proc = gmac::Process::getInstance();
    gmac::Mode *srcMode = proc.owner(buf);

    if(srcMode == NULL) return __libc_fwrite(buf, size, nmemb, stream);

	gmac::enterGmac();

    gmac::trace::Thread::io();
    gmacError_t err;
    size_t n = size * nmemb;
    size_t ret = 0;

    off_t  off  = 0;
    size_t bufferSize = paramPageSize > size ? paramPageSize : size;
    gmac::IOBuffer *buffer = proc.createIOBuffer(bufferSize);

    gmac::memory::Manager &manager = gmac::memory::Manager::getInstance();

    size_t left = n;
    buffer->lock();
    while (left != 0) {
        size_t bytes = left < bufferSize ? left : bufferSize;
        gmac::util::Logger::TRACE("Filling I/O buffer from device %p with %zd bytes (%zd)", (const char *)buf + off, bytes, size);
        err = manager.toIOBuffer(*buffer, (const char *)buf + off, bytes);
        gmac::util::Logger::ASSERTION(err == gmacSuccess);
        err = buffer->wait();
        gmac::util::Logger::ASSERTION(err == gmacSuccess);

        size_t __ret = __libc_fwrite(buffer->addr(), size, bytes/size, stream);
        ret += __ret;
        
        left -= bytes;
        off  += (off_t) bytes;
    }
    proc.destroyIOBuffer(buffer);
    gmac::trace::Thread::resume();
	gmac::exitGmac();

    return ret;
}

void stdcIoInit(void)
{
	LOAD_SYM(__libc_fread, fread);
	LOAD_SYM(__libc_fwrite, fwrite);
}
