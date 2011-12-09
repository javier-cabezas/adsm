#ifndef GMAC_HAL_EVENT_IMPL_H_
#define GMAC_HAL_EVENT_IMPL_H_

namespace __impl { namespace hal { namespace detail {

template <typename I>
inline
_event_t<I>::_event_t(bool async, type t, context_parent_t &context) :
    context_(context),
    async_(async),
    synced_(async? false: true),
    type_(t),
    state_(None),
    err_(gmacSuccess)
{
}

template <typename I>
inline
_event_t<I>::~_event_t()
{
}

template <typename I>
inline
typename I::context &
_event_t<I>::get_context()
{
    return context_;
}

template <typename I>
inline
gmacError_t
_event_t<I>::sync()
{
    if (err_ == gmacSuccess) {
        // Execute triggers
        exec_triggers();
    }

    return err_;
}

template <typename I>
inline
typename _event_t<I>::type
_event_t<I>::get_type() const
{
    return type_;
}

#if 0
template <typename I>
inline
typename _event_t<I>::state
_event_t<I>::get_state()
{
    return state_;
}
#endif

template <typename I>
inline
hal::time_t
_event_t<I>::get_time_queued() const
{
    return timeQueued_;
}


template <typename I>
inline
hal::time_t
_event_t<I>::get_time_submit() const
{
    return timeSubmit_;
}

template <typename I>
inline
hal::time_t
_event_t<I>::get_time_start() const
{
    return timeStart_;
}

template <typename I>
inline
hal::time_t
_event_t<I>::get_time_end() const
{
    return timeEnd_;
}

#if 0
template <typename I>
inline
void
_event_t<I>::set_state(state s)
{
    state_ = s;
}

template <typename I>
inline
void
_event_t<I>::set_synced(bool synced)
{
    synced_ = synced;
}
#endif

template <typename I>
inline
bool
_event_t<I>::is_synced() const
{
    return synced_;
}

} // namespace detail

}}

#endif /* GMAC_HAL_TYPES_IMPL_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
