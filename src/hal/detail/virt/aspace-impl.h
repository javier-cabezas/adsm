#ifndef GMAC_HAL_DETAIL_VIRT_ASPACE_IMPL_H_
#define GMAC_HAL_DETAIL_VIRT_ASPACE_IMPL_H_

namespace __impl { namespace hal { namespace detail {

namespace virt {

inline
buffer::buffer(size_t size, aspace &as) :
    aspace_(as),
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
_event *
queue_event::pop()
{
    _event *ret = NULL;

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
queue_event::push(_event &event)
{
	Lock::lock();
    Parent::push(&event);
    Lock::unlock();
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
aspace::aspace(phys::processing_unit &pu, phys::aspace &pas, gmacError_t &err) :
    pu_(pu),
    pas_(pas),
    nBuffersIn_(0),
    nBuffersOut_(0)
{
    err = gmacSuccess;
}

inline
aspace::~aspace()
{
}

inline
phys::processing_unit &
aspace::get_processing_unit()
{
    return pu_;
}

inline
const phys::processing_unit &
aspace::get_processing_unit() const
{
    return pu_;
}

inline
phys::aspace &
aspace::get_paspace()
{
    return pas_;
}

inline
const phys::aspace &
aspace::get_paspace() const
{
    return pas_;
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

} // namespace virt

}}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
