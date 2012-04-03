#ifndef GMAC_HAL_CUDA_CONTEXT_IMPL_H_
#define GMAC_HAL_CUDA_CONTEXT_IMPL_H_

#include "trace/logger.h"

namespace __impl { namespace hal { namespace cuda { namespace virt {

inline
buffer::buffer(host_ptr addr, size_t size, aspace &as) :
    aspace_(as),
    addr_(addr),
    size_(size)
{
}

inline
aspace &
buffer::get_aspace()
{
    return aspace_;
}

inline
const aspace &
buffer::get_aspace() const
{
    return aspace_;
}

inline void
buffer::set_event(event_ptr event)
{
    event_ = event;
}

inline
gmacError_t
buffer::wait()
{
    gmacError_t ret = gmacSuccess;
    if (event_) {
        ret = event_->sync();
    }

    return ret;
}

inline
size_t
buffer::get_size() const
{
    return size_;
}

inline
queue_event::queue_event() :
    Lock("queue_event")
{
}

inline
_event_t *
queue_event::pop()
{
    _event_t *ret = NULL;

    Lock::lock();
    if (Parent::size() > 0) {
        ret = Parent::front();
        Parent::pop();
    }
    Lock::unlock();

    return ret;
}

inline
void
queue_event::push(_event_t &event)
{
	Lock::lock();
    Parent::push(&event);
    Lock::unlock();
}

inline
void
aspace::put_input_buffer(buffer &buffer)
{
    mapUsedBuffersIn_.remove(&buffer, buffer.get_size());
    mapFreeBuffersIn_.push(&buffer, buffer.get_size());
}

inline
void
aspace::put_output_buffer(buffer &buffer)
{
    mapUsedBuffersOut_.remove(&buffer, buffer.get_size());
    mapFreeBuffersOut_.push(&buffer, buffer.get_size());
}

inline
host_ptr
aspace::get_memory(size_t size)
{
    host_ptr mem = (host_ptr) mapMemory_.pop(size);

    if (mem == NULL) {
        mem = (host_ptr) malloc(size);
    }

    return mem;
}

inline void
aspace::put_memory(void *ptr, size_t size)
{
    mapMemory_.push(ptr, size);
}

inline
host_ptr
buffer::get_addr()
{
    return addr_;
}

#if 0
inline
hal::ptr
buffer_t::get_device_addr()
{
    return get_aspace().get_device_addr_from_pinned(addr_);
}
#endif

inline
void
aspace::set()
{
    CUresult ret = cuCtxSetCurrent(this->context_);
    ASSERTION(ret == CUDA_SUCCESS);
}

#if 0
inline
hal::ptr
aspace::get_device_addr_from_pinned(host_ptr addr)
{
    hal::ptr ret;
    set();

    CUdeviceptr ptr;
    CUresult res = cuMemHostGetDevicePointer(&ptr, addr, 0);
    if (res == CUDA_SUCCESS) {
        ret = hal::ptr(hal::ptr::backend_ptr(ptr), this);
    }

    return ret;
}
#endif

}}}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
