#ifndef GMAC_HAL_STREAM_IMPL_H_
#define GMAC_HAL_STREAM_IMPL_H_

namespace __impl { namespace hal { namespace detail {

template <typename I>
inline
stream_t<I>::stream_t(context_parent_t &context) :
#if 0
    lockBuffer_("lock_buffer"),
    buffer_(NULL),
#endif
    gmac::util::spinlock<stream_t<I> >("stream_t"),
    context_(context)
{
}

template <typename I>
inline
typename stream_t<I>::context_parent_t &
stream_t<I>::get_context()
{
    return context_;
}

template <typename I>
inline
void
stream_t<I>::set_last_event(typename I::event_ptr event)
{
    this->lock();
    lastEvent_ = event;
    this->unlock();
}

template <typename I>
inline
typename I::event_ptr
stream_t<I>::get_last_event()
{
    typename I::event_ptr ret;
    this->lock();
    ret = lastEvent_;
    this->unlock();
    return ret;
}

} // namespace detail

}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
