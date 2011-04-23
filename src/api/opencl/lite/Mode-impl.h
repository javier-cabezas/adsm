#ifndef GMAC_API_OPENCL_LITE_MODE_IMPL_H_
#define GMAC_API_OPENCL_LITE_MODE_IMPL_H_

#include "core/IOBuffer.h"

namespace __impl { namespace opencl { namespace lite {

inline
Mode::Mode(cl_context ctx) :
    context_(ctx)
{}

inline
Mode::~Mode()
{}

inline
gmacError_t Mode::hostAlloc(hostptr_t &, size_t)
{
    FATAL("Host Memory allocation not supported in GMAC/Lite");
    return gmacErrorUnknown;
}

inline
gmacError_t Mode::hostFree(hostptr_t)
{
    FATAL("Host Memory release not supported in GMAC/Lite");
    return gmacErrorUnknown;
}

inline
accptr_t Mode::hostMapAddr(const hostptr_t)
{
    FATAL("Host Memory translation is not supported in GMAC/Lite");
    return accptr_t(0);
}

inline
core::IOBuffer &Mode::createIOBuffer(size_t size)
{
    IOBuffer *ret;
    void *addr = ::malloc(size);
    ret = new IOBuffer(*this, hostptr_t(addr), size, false);
    return *ret;
}

inline
void Mode::destroyIOBuffer(core::IOBuffer &buffer)
{
    ::free(buffer.addr());
    delete &buffer;
}


inline
gmacError_t Mode::bufferToAccelerator(accptr_t dst, core::IOBuffer &buffer, size_t len, size_t off)
{
    TRACE(LOCAL,"Copy %p to device %p ("FMT_SIZE" bytes)", buffer.addr(), dst.base_, len);
    gmacError_t ret = gmacSuccess;
    return ret;
}

inline
gmacError_t Mode::acceleratorToBuffer(core::IOBuffer &buffer, const accptr_t src, size_t len, size_t off)
{
    TRACE(LOCAL,"Copy %p to host %p ("FMT_SIZE" bytes)", src.base_, buffer.addr(), len);
    gmacError_t ret = gmacSuccess;
    return ret;
}

inline
cl_command_queue Mode::eventStream()
{
    return stream_;
}

inline
gmacError_t Mode::waitForEvent(cl_event event)
{
    //TODO: implement waiting for events
    return gmacErrorUnknown;
}

inline
gmacError_t Mode::eventTime(uint64_t &t, cl_event start, cl_event end)
{
    gmacError_t ret = gmacSuccess;
    return ret; 
}

}}}

#endif
