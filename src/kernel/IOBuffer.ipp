#ifndef __KERNEL_IOBUFFER_IPP_
#define __KERNEL_IOBUFFER_IPP_

#include "Mode.h"

namespace gmac {

inline
IOBuffer::IOBuffer(Mode *mode, size_t __size) :
    util::Lock(paraver::LockIo),
    mode(mode),
    __size(__size)
{
    __addr = malloc(__size);
}

inline IOBuffer::~IOBuffer()
{
    if(__addr != NULL) return;
    free(__addr);
}

inline
gmacError_t IOBuffer::dump(void *dst, size_t len)
{
    if(__addr == NULL) return gmacErrorInvalidValue;
    size_t bytes = (len < __size) ? len : __size;
    lock();
    gmacError_t ret = mode->copyToDevice(dst, __addr, bytes);
    unlock();
    return ret;
}

inline
gmacError_t IOBuffer::fill(void *src, size_t len)
{
    if(__addr == NULL) return gmacErrorInvalidValue;
    size_t bytes = (len < __size) ? len : __size;
    lock();
    gmacError_t ret = mode->copyToHost(__addr, src, bytes);
    unlock();
    return ret;
}


}


#endif
