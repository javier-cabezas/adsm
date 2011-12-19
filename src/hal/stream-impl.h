#ifndef GMAC_HAL_STREAM_IMPL_H_
#define GMAC_HAL_STREAM_IMPL_H_

namespace __impl { namespace hal { namespace detail {

template <typename B, typename I>
inline
stream_t<B, I>::stream_t(typename B::stream stream, context_parent_t &context) :
#if 0
    lockBuffer_("lock_buffer"),
    buffer_(NULL),
#endif
    gmac::util::spinlock<stream_t<B, I> >("stream_t"),
    stream_(stream),
    context_(context)
{
}

template <typename B, typename I>
inline
typename stream_t<B, I>::context_parent_t &
stream_t<B, I>::get_context()
{
    return context_;
}

template <typename B, typename I>
inline
typename B::stream &
stream_t<B, I>::operator()()
{
    return stream_;
}

template <typename B, typename I>
inline
const typename B::stream &
stream_t<B, I>::operator()() const
{
    return stream_;
}

template <typename B, typename I>
inline
void
stream_t<B, I>::set_last_event(typename I::event_ptr event)
{
    this->lock();
    lastEvent_ = event;
    this->unlock();
}

template <typename B, typename I>
inline
typename I::event_ptr
stream_t<B, I>::get_last_event()
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
