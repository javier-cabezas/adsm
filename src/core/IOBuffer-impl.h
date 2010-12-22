#ifndef GMAC_CORE_IOBUFFER_IMPL_H_
#define GMAC_CORE_IOBUFFER_IMPL_H_

namespace __impl { namespace core {

inline
IOBuffer::IOBuffer(void *addr, size_t size) :
        gmac::util::Lock("IOBuffer"), addr_(addr), size_(size), state_(Idle)
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
    gmac::util::Lock::lock();
}

inline void
IOBuffer::unlock()
{
    gmac::util::Lock::unlock();
}

inline IOBuffer::State
IOBuffer::state() const
{
    return state_;
}

}}

#endif
