#include <cstdio>
#include <errno.h>

#include "core/address_space.h"
#include "core/vdevice.h"

#include "libs/common.h"

#include "memory/Manager.h"

#include "trace/Tracer.h"

#include "util/loader.h"
#include "util/Logger.h"

#include "stdc.h"

using namespace __impl::core;
using namespace __impl::memory;
using namespace __impl::util;

SYM(size_t, __libc_fread, void *, size_t, size_t, FILE *);
SYM(size_t, __libc_fwrite, const void *, size_t, size_t, FILE *);

class GMAC_LOCAL stdc_input :
    public __impl::hal::device_input {
    FILE *file_;
    size_t sizeElem_;

    ssize_t result_;

public:
    stdc_input(FILE *file, size_t sizeElem) :
        file_(file),
        sizeElem_(sizeElem)
    {
    }

    bool read(void *ptr, size_t count)
    {
        bool ok;

        size_t res = fread(ptr, sizeElem_, count/sizeElem_, file_);
        ok = (res == count);
        result_ += res;

        return ok;
    }

    ssize_t get_result() const
    {
        return result_;
    }
};

class GMAC_LOCAL stdc_output :
    public __impl::hal::device_output {
    FILE *file_;
    size_t sizeElem_;

    ssize_t result_;

public:
    stdc_output(FILE *file, size_t sizeElem) :
        file_(file),
        sizeElem_(sizeElem)
    {
    }

    bool write(void *ptr, size_t count)
    {
        bool ok;

        size_t res = fwrite(ptr, sizeElem_, count/sizeElem_, file_);
        ok = (res == count);
        result_ += res;

        return ok;
    }

    ssize_t get_result() const
    {
        return result_;
    }
};

/* Library wrappers */

#ifdef __cplusplus
extern "C"
#endif
size_t SYMBOL(fread)(void *buf, size_t size, size_t nmemb, FILE *stream)
{
	if(__libc_fread == NULL) stdcIoInit();
	if((inGmac() == 1) ||
       (size * nmemb == 0)) return __libc_fread(buf, size, nmemb, stream);

    enterGmac();
    Manager &manager = getManager();
    smart_ptr<address_space>::shared aspaceDst = manager.owner(hostptr_t(buf));

    if(aspaceDst == NULL) {
        exitGmac();
        return  __libc_fread(buf, size, nmemb, stream);
    }

	gmac::trace::SetThreadState(gmac::trace::IO);

    stdc_input op(stream, size);

    manager.from_io_device(aspaceDst, hostptr_t(buf), op, size * nmemb);
    ssize_t ret = op.get_result();

	gmac::trace::SetThreadState(gmac::trace::Running);
	exitGmac();

    return ret;
}

#ifdef __cplusplus
extern "C"
#endif
size_t SYMBOL(fwrite)(const void *buf, size_t size, size_t nmemb, FILE *stream)
{
    if(__libc_fwrite == NULL) stdcIoInit();
	if((inGmac() == 1) ||
       (size * nmemb == 0)) return __libc_fwrite(buf, size, nmemb, stream);

	enterGmac();
    Manager &manager = getManager();
    smart_ptr<address_space>::shared aspaceSrc = manager.owner(hostptr_t(buf));

    if(aspaceSrc == NULL) {
        exitGmac();
        return __libc_fwrite(buf, size, nmemb, stream);
    }

	gmac::trace::SetThreadState(gmac::trace::IO);

    stdc_output op(stream, size);

    manager.to_io_device(op, aspaceSrc, hostptr_t(buf), size * nmemb);
    ssize_t ret = op.get_result();

	gmac::trace::SetThreadState(gmac::trace::Running);
	exitGmac();

    return ret;
}

void stdcIoInit(void)
{
	LOAD_SYM(__libc_fread, fread);
	LOAD_SYM(__libc_fwrite, fwrite);
}
