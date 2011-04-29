
#include "core/IOBuffer.h"
#include "core/hpe/Mode.h"
#include "core/hpe/Process.h"

#include "memory/Manager.h"

#include "hpe/init.h"

#include "os/posix/loader.h"

#include <unistd.h>
#include <cstdio>
#include <stdint.h>
#include <dlfcn.h>

#include <errno.h>

#include "posix.h"

using __impl::core::IOBuffer;
using __impl::core::Mode;
using __impl::core::Process;

using __impl::memory::Manager;

using __impl::util::params::ParamBlockSize;

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
	if(inGmac() == 1 || count == 0) return __libc_read(fd, buf, count);

    enterGmac();
    Process &proc = Process::getInstance();
    Mode *dstMode = proc.owner(hostptr_t(buf));

    if(dstMode == NULL) {
        exitGmac();
        return __libc_read(fd, buf, count);
    }

	gmac::trace::SetThreadState(gmac::trace::IO);
    gmacError_t err;
    size_t ret = 0;
    size_t bufferSize = ParamBlockSize > count ? ParamBlockSize : count;
    Mode &mode = __impl::core::hpe::Mode::getCurrent();
    IOBuffer *buffer1 = &mode.createIOBuffer(bufferSize);
    IOBuffer *buffer2 = NULL;
    if (count > buffer1->size()) {
        buffer2 = &mode.createIOBuffer(bufferSize);
    }

    Manager &manager = Manager::getInstance();
    IOBuffer *active  = buffer1;
    IOBuffer *passive = buffer2;

    size_t left = count;
    off_t  off  = 0;
    while (left != 0) {
        err = active->wait();
        ASSERTION(err == gmacSuccess);
        size_t bytes= left < active->size()? left: active->size();
        ret += __libc_read(fd, active->addr(), bytes);
        ret = manager.fromIOBuffer(mode, hostptr_t(buf) + off, *active, 0, bytes);
        ASSERTION(ret == gmacSuccess);

        left -= bytes;
        off  += bytes;
        IOBuffer *tmp = active;
        active = passive;
        passive = tmp;
    }
    err = passive->wait();
    ASSERTION(err == gmacSuccess);
    mode.destroyIOBuffer(*buffer1);
    if (buffer2 != NULL) {
        mode.destroyIOBuffer(*buffer2);
    }
	gmac::trace::SetThreadState(gmac::trace::Running);
	exitGmac();

    return ret;
}

#ifdef __cplusplus
extern "C"
#endif
ssize_t write(int fd, const void *buf, size_t count)
{
	if(__libc_read == NULL) posixIoInit();
	if(inGmac() == 1 || count == 0) return __libc_write(fd, buf, count);

	enterGmac();
    Process &proc = Process::getInstance();
    Mode *srcMode = proc.owner(hostptr_t(buf));

    if(srcMode == NULL) {
        exitGmac();
        return __libc_write(fd, buf, count);
    }

	gmac::trace::SetThreadState(gmac::trace::IO);
    gmacError_t err;
    size_t ret = 0;

    off_t  off  = 0;
    size_t bufferSize = ParamBlockSize > count ? ParamBlockSize : count;
    Mode &mode = __impl::core::hpe::Mode::getCurrent();
    IOBuffer *buffer1 = &mode.createIOBuffer(bufferSize);
    IOBuffer *buffer2 = NULL;
    if (count > buffer1->size()) {
        buffer2 = &mode.createIOBuffer(bufferSize);
    }

    Manager &manager = Manager::getInstance();
    IOBuffer *active  = buffer1;
    IOBuffer *passive = buffer2;

    size_t left = count;

    size_t bytesActive = left < active->size() ? left : active->size();
    err = manager.toIOBuffer(mode, *active, 0, hostptr_t(buf) + off, bytesActive);
    ASSERTION(err == gmacSuccess);
    size_t bytesPassive = 0;

    do {
        left -= bytesActive;
        off  += bytesActive;

        if (left > 0) {
            bytesPassive = left < passive->size()? left : passive->size();
            err = manager.toIOBuffer(__impl::core::hpe::Mode::getCurrent(), *passive, 0, hostptr_t(buf) + off, bytesPassive);
            ASSERTION(err == gmacSuccess);
        }

        err = active->wait();
        ASSERTION(err == gmacSuccess);

        ret += __libc_write(fd, active->addr(), bytesActive);

        size_t bytesTmp = bytesActive;
        bytesActive = bytesPassive;
        bytesPassive = bytesTmp;
        
        IOBuffer *tmp = active;
        active = passive;
        passive = tmp;
    } while (left != 0);
    ASSERTION(err == gmacSuccess);
    mode.destroyIOBuffer(*buffer1);
    if (buffer2 != NULL) {
        mode.destroyIOBuffer(*buffer2);
    }
	gmac::trace::SetThreadState(gmac::trace::Running);
	exitGmac();

    return ret;
}
