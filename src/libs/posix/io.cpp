#if !defined(_MSC_VER)
#include <unistd.h>
#include <stdint.h>
#include <dlfcn.h>
#endif

#include <cstdio>
#include <errno.h>

#include "core/address_space.h"
#include "core/vdevice.h"

#include "libs/common.h"
#include "memory/Manager.h"
#include "util/loader.h"

#include "posix.h"

using namespace __impl::core;
using namespace __impl::memory;
using namespace __impl::util;
using __impl::util::params::ParamBlockSize;

SYM(ssize_t, __libc_read, int, void *, size_t);
SYM(ssize_t, __libc_write, int, const void *, size_t);

class GMAC_LOCAL posix_input :
    public __impl::hal::device_input {
    int fd_;

    ssize_t result_;

public:
    posix_input(int fd) :
        fd_(fd)
    {
    }

    bool read(void *ptr, size_t count)
    {
        bool ok;

        ssize_t res = ::read(fd_, ptr, count);
        ok = (res == ssize_t(count));
        if (res < 0) {
            result_ = res;
        } else {
            result_ += res;
        }

        return ok;
    }

    ssize_t get_result() const
    {
        return result_;
    }
};

class GMAC_LOCAL posix_output :
    public __impl::hal::device_output {
    int fd_;

    ssize_t result_;

public:
    posix_output(int fd) :
        fd_(fd)
    {
    }

    bool write(void *ptr, size_t count)
    {
        bool ok;

        ssize_t res = ::write(fd_, ptr, count);
        ok = (res == ssize_t(count));
        if (res < 0) {
            result_ = res;
        } else {
            result_ += res;
        }

        return ok;
    }

    ssize_t get_result() const
    {
        return result_;
    }
};

/* System call wrappers */

#ifdef __cplusplus
extern "C"
#endif
ssize_t SYMBOL(read)(int fd, void *buf, size_t count)
{
	if(__libc_read == NULL) posixIoInit();
	if(inGmac() == 1 || count == 0)
        return __libc_read(fd, buf, count);

    enterGmac();
    Manager &manager = getManager();
    address_space_ptr aspaceDst = manager.owner(hostptr_t(buf));

    if(aspaceDst == NULL) {
        exitGmac();
        return __libc_read(fd, buf, count);
    }

	gmac::trace::SetThreadState(gmac::trace::IO);

    posix_input op(fd);

    manager.from_io_device(aspaceDst, hostptr_t(buf), op, count);
    ssize_t ret = op.get_result();
#if 0
    gmacError_t err;
    ssize_t ret = 0;
    size_t bufferSize = ParamBlockSize > count ? ParamBlockSize : count;
    io_buffer *buffer1 = aspaceDst->create_io_buffer(bufferSize, GMAC_PROT_READ);
    io_buffer *buffer2 = NULL;
    if (count > buffer1->size()) {
        buffer2 = aspaceDst->create_io_buffer(bufferSize, GMAC_PROT_READ);
    }

    io_buffer *active  = buffer1;
    io_buffer *passive = buffer2;

    size_t left = count;
    size_t  off  = 0;
    while (left != 0) {
        err = active->wait();
        ASSERTION(err == gmacSuccess);
        size_t bytes= left < active->size()? left: active->size();
        ret += __libc_read(fd, active->addr(), bytes);
        ret = manager.fromIOBuffer(aspaceDst, hostptr_t(buf) + off, *active, 0, bytes);
        ASSERTION(ret == gmacSuccess);

        left -= bytes;
        off  += bytes;
        io_buffer *tmp = active;
        active = passive;
        passive = tmp;
    }
    err = passive->wait();
    ASSERTION(err == gmacSuccess);
    aspaceDst->destroy_io_buffer(*buffer1);
    if (buffer2 != NULL) {
        aspaceDst->destroy_io_buffer(*buffer2);
    }
#endif
	gmac::trace::SetThreadState(gmac::trace::Running);
	exitGmac();

    return ret;
}

#ifdef __cplusplus
extern "C"
#endif
ssize_t SYMBOL(write)(int fd, const void *buf, size_t count)
{
	if(__libc_read == NULL) posixIoInit();
	if(inGmac() == 1 || count == 0)
        return __libc_write(fd, buf, count);

	enterGmac();
    Manager &manager = getManager();
    address_space_ptr aspaceSrc = manager.owner(hostptr_t(buf));

    if(aspaceSrc == NULL) {
        exitGmac();
        return __libc_write(fd, buf, count);
    }

	gmac::trace::SetThreadState(gmac::trace::IO);

    posix_output op(fd);

    manager.to_io_device(op, aspaceSrc, hostptr_t(buf), count);
    ssize_t ret = op.get_result();
#if 0
    gmacError_t err;
    ssize_t ret = 0;

    size_t off  = 0;
    size_t bufferSize = ParamBlockSize > count ? ParamBlockSize : count;
    io_buffer *buffer1 = aspaceSrc->create_io_buffer(bufferSize, GMAC_PROT_READ);
    io_buffer *buffer2 = NULL;
    if (count > buffer1->size()) {
        buffer2 = aspaceSrc->create_io_buffer(bufferSize, GMAC_PROT_READ);
    }

    io_buffer *active  = buffer1;
    io_buffer *passive = buffer2;

    size_t left = count;

    size_t bytesActive = left < active->size() ? left : active->size();
    err = manager.toIOBuffer(aspaceSrc, *active, 0, hostptr_t(buf) + off, bytesActive);
    ASSERTION(err == gmacSuccess);
    size_t bytesPassive = 0;

    do {
        left -= bytesActive;
        off  += bytesActive;

        if (left > 0) {
            bytesPassive = left < passive->size()? left : passive->size();
            err = manager.toIOBuffer(aspaceSrc, *passive, 0, hostptr_t(buf) + off, bytesPassive);
            ASSERTION(err == gmacSuccess);
        }
        err = active->wait();
        ASSERTION(err == gmacSuccess);

        ret += __libc_write(fd, active->addr(), bytesActive);

        size_t bytesTmp = bytesActive;
        bytesActive = bytesPassive;
        bytesPassive = bytesTmp;
        
        io_buffer *tmp = active;
        active = passive;
        passive = tmp;
    } while (left != 0);
    ASSERTION(err == gmacSuccess);
    aspaceSrc->destroy_io_buffer(*buffer1);
    if (buffer2 != NULL) {
        aspaceSrc->destroy_io_buffer(*buffer2);
    }
#endif
	gmac::trace::SetThreadState(gmac::trace::Running);
	exitGmac();

    return ret;
}

void posixIoInit(void)
{
	TRACE(GLOBAL, "Overloading I/O POSIX functions");
	LOAD_SYM(__libc_read, read);
	LOAD_SYM(__libc_write, write);
}
