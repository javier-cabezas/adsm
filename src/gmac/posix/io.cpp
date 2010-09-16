#include <gmac/init.h>
#include <os/loader.h>
#include <memory/Manager.h>
#include <kernel/Mode.h>
#include <kernel/IOBuffer.h>
#include <trace/Thread.h>

#include <unistd.h>
#include <cstdio>
#include <stdint.h>
#include <dlfcn.h>

#include <errno.h>

#include "posix.h"

SYM(ssize_t, __libc_read, int, void *, size_t);
SYM(ssize_t, __libc_write, int, const void *, size_t);

void posixIoInit(void)
{
	gmac::util::Logger::TRACE("Overloading I/O POSIX functions");
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
    gmac::Mode *dstMode = gmac::proc->owner(buf);

    if(dstMode == NULL) {
        gmac::exitGmac();
        return __libc_read(fd, buf, count);
    }

    gmac::trace::Thread::io();

    gmacError_t err;
    size_t ret = 0;

    gmac::IOBuffer *buffer = gmac::proc->createIOBuffer(paramPageSize);

    size_t left = count;
    off_t  off  = 0;
    while (left != 0) {
        size_t bytes= left < buffer->size()? left: buffer->size();
        ret += __libc_read(fd, buffer->addr(), bytes);
        ret = gmac::manager->fromIOBuffer((char *)buf + off, *buffer, bytes);
        gmac::util::Logger::ASSERTION(ret == gmacSuccess);

        left -= bytes;
        off  += bytes;
    }
    gmac::proc->destroyIOBuffer(buffer);
    gmac::trace::Thread::resume();
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
    gmac::Mode *srcMode = gmac::proc->owner(buf);

    if(srcMode == NULL) {
        gmac::exitGmac();
        return __libc_write(fd, buf, count);
    }

    gmac::trace::Thread::io();

    gmacError_t err;
    size_t ret = 0;

    off_t  off  = 0;
    gmac::IOBuffer *buffer = gmac::proc->createIOBuffer(paramPageSize);

    size_t left = count;
    while (left != 0) {
        size_t bytes = left < buffer->size() ? left : buffer->size();
        err = gmac::manager->toIOBuffer(*buffer, (char *)buf + off, bytes);
        gmac::util::Logger::ASSERTION(err == gmacSuccess);
        ret += __libc_write(fd, buffer->addr(), bytes);

        left -= bytes;
        off  += bytes;
    }
    gmac::proc->destroyIOBuffer(buffer);
    gmac::trace::Thread::resume();
	gmac::exitGmac();

    return ret;
}
