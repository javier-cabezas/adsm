#ifndef GMAC_CORE_IOBUFFER_IMPL_H_
#define GMAC_CORE_IOBUFFER_IMPL_H_

#include "util/Logger.h"

namespace __impl { namespace core {

inline
io_buffer::io_buffer(hal::context_t &context, size_t size, GmacProtection prot) :
    size_(size), state_(Idle), prot_(prot)
{
    gmacError_t res;

    buffer_ = context.alloc_buffer(size, prot, res);

    if (res == gmacSuccess) {
        addr_ = buffer_->get_addr();
        async_ = true;
    } else {
        addr_ = malloc(size);
        CFATAL(addr_ != NULL, "Error allocating memory");
        async_ = false;
    }
}

inline
io_buffer::~io_buffer()
{
    if (buffer_ != NULL) {
        buffer_->get_context().free_buffer(*buffer_);
    }
}

inline
hal::buffer_t &
io_buffer::get_buffer()
{
    return *buffer_;
}

inline
const hal::buffer_t &
io_buffer::get_buffer() const
{
    return *buffer_;
}

inline uint8_t *
io_buffer::addr() const
{
    return static_cast<uint8_t *>(addr_);
}

inline uint8_t *
io_buffer::end() const
{
    return addr() + size_;
}

inline size_t
io_buffer::size() const
{
    return size_;
}

inline bool
io_buffer::async() const
{
    return async_;
}

inline io_buffer::State
io_buffer::state() const
{
    return state_;
}

inline gmacError_t
io_buffer::wait()
{
    ASSERTION(state_ == Idle || event_.is_valid());

    gmacError_t ret = gmacSuccess;

    if (state_ != Idle) {
        ret = event_.sync();
#if 0
        ASSERTION(mode_ != NULL);
        ASSERTION(started_ == true);
        CUevent start = it->second.first;
        CUevent end   = it->second.second;
        trace::SetThreadState(trace::Wait);
        ret = mode_->waitForEvent(end, internal);
        trace::SetThreadState(trace::Running);
        if(state_ == ToHost) DataCommToHost(*mode_, start, end, xfer_);
        else if(state_ == ToAccelerator) DataCommToAccelerator(*mode_, start, end, xfer_);
        TRACE(LOCAL,"Buffer %p goes Idle", this);
        state_ = Idle;
        mode_  = NULL;
        started_ = false;
#endif
        state_ = Idle;

        // TODO: destroy event
    }

    return ret;
}

inline
void
io_buffer::to_host(hal::event_t event)
{
    ASSERTION(state_ == Idle);
    ASSERTION(event_.is_valid() == false);

    state_ = ToHost;
    event_ = event;
}

inline
void
io_buffer::to_device(hal::event_t event)
{
    ASSERTION(state_ == Idle);
    ASSERTION(event_.is_valid() == false);

    state_ = ToAccelerator;
    event_ = event;
}

inline
GmacProtection
io_buffer::get_protection() const
{
    return prot_;
}

}}

#endif
