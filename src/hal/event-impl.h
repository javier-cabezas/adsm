#ifndef GMAC_HAL_EVENT_IMPL_H_
#define GMAC_HAL_EVENT_IMPL_H_

namespace __impl { namespace hal { namespace detail {

template <typename D, typename B, typename I>
inline
event_t<D, B, I>::event_t(bool async, type t, context_parent_t &context) :
    context_(context),
    isAsynchronous_(async),
    synced_(async? false: true),
    type_(t),
    state_(None),
    err_(gmacSuccess)
{
}

template <typename D, typename B, typename I>
inline
event_t<D, B, I>::~event_t()
{
}

template <typename D, typename B, typename I>
inline
typename I::context &
event_t<D, B, I>::get_context()
{
    return context_;
}

template <typename D, typename B, typename I>
inline
gmacError_t
event_t<D, B, I>::sync()
{
    return err_;
}

template <typename D, typename B, typename I>
inline
typename event_t<D, B, I>::type
event_t<D, B, I>::get_type() const
{
    return type_;
}

#if 0
template <typename D, typename B, typename I>
inline
typename event_t<D, B, I>::state
event_t<D, B, I>::get_state()
{
    return state_;
}
#endif

template <typename D, typename B, typename I>
inline
hal::time_t
event_t<D, B, I>::get_time_queued() const
{
    return timeQueued_;
}


template <typename D, typename B, typename I>
inline
hal::time_t
event_t<D, B, I>::get_time_submit() const
{
    return timeSubmit_;
}

template <typename D, typename B, typename I>
inline
hal::time_t
event_t<D, B, I>::get_time_start() const
{
    return timeStart_;
}

template <typename D, typename B, typename I>
inline
hal::time_t
event_t<D, B, I>::get_time_end() const
{
    return timeEnd_;
}

#if 0
template <typename D, typename B, typename I>
inline
void
event_t<D, B, I>::set_state(state s)
{
    state_ = s;
}

template <typename D, typename B, typename I>
inline
void
event_t<D, B, I>::set_synced(bool synced)
{
    synced_ = synced;
}
#endif

template <typename D, typename B, typename I>
inline
bool
event_t<D, B, I>::is_synced() const
{
    return synced_;
}

} // namespace detail

}}

#endif /* GMAC_HAL_TYPES_IMPL_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
