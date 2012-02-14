#ifndef GMAC_HAL_STREAM_IMPL_H_
#define GMAC_HAL_STREAM_IMPL_H_

namespace __impl { namespace hal { namespace detail {

inline
stream::stream(aspace &as) :
#if 0
    lockBuffer_("lock_buffer"),
    buffer_(NULL),
#endif
    gmac::util::spinlock<stream>("stream"),
    aspace_(as)
{
}

inline
aspace &
stream::get_aspace()
{
    return aspace_;
}

inline
void
stream::set_last_event(event_ptr event)
{
    this->lock();
    lastEvent_ = event;
    this->unlock();
}

inline
event_ptr
stream::get_last_event()
{
    event_ptr ret;
    this->lock();
    ret = lastEvent_;
    this->unlock();
    return ret;
}

} // namespace detail

}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
