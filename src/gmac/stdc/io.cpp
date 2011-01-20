#include <cstdio>
#include <errno.h>

#if defined(POSIX)
#include "os/posix/loader.h"
#elif defined(WINDOWS)
#include "os/windows/loader.h"
#endif

#include "gmac/init.h"
#include "memory/Manager.h"
#include "core/IOBuffer.h"
#include "core/Process.h"
#include "core/Mode.h"
#include "trace/Tracer.h"
#include "util/Logger.h"

#include "stdc.h"

using __impl::core::IOBuffer;
using __impl::core::Mode;
using __impl::core::Process;

using __impl::memory::Manager;

SYM(size_t, __libc_fread, void *, size_t, size_t, FILE *);
SYM(size_t, __libc_fwrite, const void *, size_t, size_t, FILE *);

#ifdef __cplusplus
extern "C"
#endif
size_t SYMBOL(fread)(void *buf, size_t size, size_t nmemb, FILE *stream)
{
	if(__libc_fread == NULL) stdcIoInit();
	if(gmac::inGmac() == 1) return __libc_fread(buf, size, nmemb, stream);

    Process &proc = Process::getInstance();
    Mode *dstMode = proc.owner(hostptr_t(buf), size);

    if(dstMode == NULL) return  __libc_fread(buf, size, nmemb, stream);

    gmac::enterGmac();
	
	gmac::trace::SetThreadState(gmac::trace::IO);
    gmacError_t err;
    size_t n = size * nmemb;
    size_t ret = 0;

    size_t off = 0;
    size_t bufferSize = paramBlockSize > size ? paramBlockSize : size;
    Mode &mode = Mode::getCurrent();
    IOBuffer *buffer = mode.createIOBuffer(bufferSize);

    Manager &manager = Manager::getInstance();
    
    size_t left = n;
    while (left != 0) {
        size_t bytes= left < buffer->size()? left: buffer->size();
        size_t elems = __libc_fread(buffer->addr(), size, bytes/size, stream);
        ASSERTION(elems * size == bytes);
		ret += elems;
        err = manager.fromIOBuffer((uint8_t *)buf + off, *buffer, 0, size * elems);
        ASSERTION(err == gmacSuccess);
        err = buffer->wait();
        ASSERTION(err == gmacSuccess);

        left -= size * elems;
        off  += size * elems;
        TRACE(GLOBAL, FMT_SIZE" of %zd bytes read", elems * size, nmemb * size);
    }
    mode.destroyIOBuffer(buffer);
	gmac::trace::SetThreadState(gmac::trace::Running);
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

    Process &proc = Process::getInstance();
    Mode *srcMode = proc.owner(hostptr_t(buf), size);

    if(srcMode == NULL) return __libc_fwrite(buf, size, nmemb, stream);

	gmac::enterGmac();

	gmac::trace::SetThreadState(gmac::trace::IO);
    gmacError_t err;
    size_t n = size * nmemb;
    size_t ret = 0;

    size_t off = 0;
    size_t bufferSize = paramBlockSize > size ? paramBlockSize : size;
    Mode &mode = Mode::getCurrent();
    IOBuffer *buffer = mode.createIOBuffer(bufferSize);

    Manager &manager = Manager::getInstance();

    size_t left = n;
    while (left != 0) {
        size_t bytes = left < buffer->size()? left : buffer->size();
        err = manager.toIOBuffer(*buffer, 0, hostptr_t(buf) + off, bytes);
        ASSERTION(err == gmacSuccess);
        err = buffer->wait();
        ASSERTION(err == gmacSuccess);

        size_t elems = __libc_fwrite(buffer->addr(), size, bytes/size, stream);
        ASSERTION(elems * size == bytes);
        ret += elems;
        
        left -= size * elems;
        off  += size * elems;

        TRACE(GLOBAL, FMT_SIZE" of "FMT_SIZE" bytes written", elems * size, nmemb * size);
    }
    mode.destroyIOBuffer(buffer);
	gmac::trace::SetThreadState(gmac::trace::Running);
	gmac::exitGmac();

    return ret;
}

void stdcIoInit(void)
{
	LOAD_SYM(__libc_fread, fread);
	LOAD_SYM(__libc_fwrite, fwrite);
}
