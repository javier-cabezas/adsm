#ifndef GMAC_CORE_IOBUFFER_IPP_
#define GMAC_CORE_IOBUFFER_IPP_

#include "trace/Function.h"

namespace gmac {

inline
IOBuffer::IOBuffer(Mode &mode, void *addr, size_t size) :
        util::Lock("IOBuffer"), addr_(addr), size_(size), state_(Idle), mode_(mode)
{
}

inline
IOBuffer::~IOBuffer()
{
}

inline uint8_t *
IOBuffer::addr() const
{
    return static_cast<uint8_t *>(addr_);
}

inline uint8_t *
IOBuffer::end() const
{
    return addr() + size_;
}

inline size_t
IOBuffer::size() const
{
    return size_;
}

inline void
IOBuffer::lock()
{
    util::Lock::lock();
}

inline void
IOBuffer::unlock()
{
    util::Lock::unlock();
}

inline IOBuffer::State
IOBuffer::state() const
{
    return state_;
}

inline void
IOBuffer::toHost()
{
    state_ = ToHost;
    trace("Buffer %p goes toHost", this); 
}

inline void
IOBuffer::toAccelerator()
{
    state_ = ToAccelerator;
    trace("Buffer %p goes toAccelerator", this);
}

inline gmacError_t
IOBuffer::wait()
{
    gmacError_t ret = gmacSuccess;
    if (state_ != Idle) {
        ret = mode_.waitForBuffer(*this);
        trace("Buffer %p goes Idle", this);
        state_ = Idle;
    }
    return ret;
}

}

#endif
