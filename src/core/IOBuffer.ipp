#ifndef GMAC_CORE_IOBUFFER_IPP_
#define GMAC_CORE_IOBUFFER_IPP_

namespace __impl { namespace core {

inline
IOBuffer::IOBuffer(void *addr, size_t size) :
        gmac::util::Lock("IOBuffer"), addr_(addr), size_(size), state_(Idle), mode_(NULL)
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

inline void
IOBuffer::toHost(Mode &mode)
{
    ASSERTION(mode_  == NULL);

    mode_  = &mode;
    state_ = ToHost;
    TRACE(LOCAL,"Buffer %p goes toHost", this); 
}

inline void
IOBuffer::toAccelerator(Mode &mode)
{
    ASSERTION(mode_  == NULL);

    mode_  = &mode;
    state_ = ToAccelerator;
    TRACE(LOCAL,"Buffer %p goes toAccelerator", this);
}

inline gmacError_t
IOBuffer::wait()
{
    gmacError_t ret = gmacSuccess;

    if (state_ != Idle) {
        ASSERTION(mode_ != NULL);
        ret = mode_->waitForBuffer(*this);
        TRACE(LOCAL,"Buffer %p goes Idle", this);
        state_ = Idle;
        mode_  = NULL;
    } else {
        ASSERTION(mode_ == NULL);
    }

    return ret;
}

}}

#endif
