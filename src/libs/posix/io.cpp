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
#include "memory/manager.h"
#include "util/loader.h"

#include "posix.h"

using namespace __impl::core;
using namespace __impl::memory;
using namespace __impl::util;

SYM(ssize_t, __libc_read, int, void *, size_t);
SYM(ssize_t, __libc_write, int, const void *, size_t);

class GMAC_LOCAL posix_input :
    public __impl::hal::device_input {
    int fd_;

    ssize_t result_;

public:
    posix_input(int fd) :
        fd_(fd),
        result_(0)
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
        fd_(fd),
        result_(0)
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
    manager &manager = get_manager();
    address_space_ptr aspaceDst = manager.get_owner(host_ptr(buf));

    if (!aspaceDst) {
        exitGmac();
        return __libc_read(fd, buf, count);
    }

	gmac::trace::SetThreadState(gmac::trace::IO);

    posix_input op(fd);

    manager.from_io_device(aspaceDst, host_ptr(buf), op, count);
    ssize_t ret = op.get_result();

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
    manager &manager = get_manager();
    address_space_ptr aspaceSrc = manager.get_owner(host_ptr(buf));

    if (!aspaceSrc) {
        exitGmac();
        return __libc_write(fd, buf, count);
    }

	gmac::trace::SetThreadState(gmac::trace::IO);

    posix_output op(fd);

    manager.to_io_device(op, aspaceSrc, host_ptr(buf), count);
    ssize_t ret = op.get_result();

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
