#include "os/posix/loader.h"
#include "core/Mode.h"
#include "core/Process.h"
#include "core/IOBuffer.h"
#include "gmac/init.h"
#include "memory/Manager.h"

#include <unistd.h>
#include <cstdio>
#include <stdint.h>
#include <dlfcn.h>

#include <errno.h>

#include "posix.h"

using __impl::core::IOBuffer;
using __impl::core::Mode;
using __impl::core::Process;

SYM(ssize_t, __libc_read, int, void *, size_t);
SYM(ssize_t, __libc_write, int, const void *, size_t);

void posixIoInit(void)
{
	TRACE(GLOBAL, "Overloading I/O POSIX functions");
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
	if(gmac::inGmac() == 1) return __libc_read(fd, buf, count);


    gmac::enterGmac();
    Process &proc = Process::getInstance();
    Mode *dstMode = proc.owner(hostptr_t(buf));

    if(dstMode == NULL) {
        gmac::exitGmac();
        return __libc_read(fd, buf, count);
    }

    gmacError_t err;
    size_t ret = 0;

    IOBuffer *buffer = Mode::getCurrent().createIOBuffer(paramBlockSize);

    gmac::memory::Manager &manager = gmac::memory::Manager::getInstance();

    size_t left = count;
    off_t  off  = 0;
    while (left != 0) {
        size_t bytes= left < buffer->size()? left: buffer->size();
        ret += __libc_read(fd, buffer->addr(), bytes);
        ret = manager.fromIOBuffer(hostptr_t(buf) + off, *buffer, 0, bytes);
        ASSERTION(ret == gmacSuccess);

        left -= bytes;
        off  += bytes;
    }
    Mode::getCurrent().destroyIOBuffer(buffer);
	gmac::exitGmac();

    return ret;
}

#ifdef __cplusplus
extern "C"
#endif
ssize_t write(int fd, const void *buf, size_t count)
{
	if(__libc_read == NULL) posixIoInit();
	if(gmac::inGmac() == 1) return __libc_write(fd, buf, count);

	gmac::enterGmac();
    Process &proc = Process::getInstance();
    Mode *srcMode = proc.owner(hostptr_t(buf));

    if(srcMode == NULL) {
        gmac::exitGmac();
        return __libc_write(fd, buf, count);
    }

    gmacError_t err;
    size_t ret = 0;

    off_t  off  = 0;
    IOBuffer *buffer = Mode::getCurrent().createIOBuffer(paramBlockSize);

    gmac::memory::Manager &manager = gmac::memory::Manager::getInstance();

    size_t left = count;
    while (left != 0) {
        size_t bytes = left < buffer->size() ? left : buffer->size();
        err = manager.toIOBuffer(*buffer, 0, hostptr_t(buf) + off, bytes);
        ASSERTION(err == gmacSuccess);
        ret += __libc_write(fd, buffer->addr(), bytes);

        left -= bytes;
        off  += bytes;
    }
    Mode::getCurrent().destroyIOBuffer(buffer);
	gmac::exitGmac();

    return ret;
}
