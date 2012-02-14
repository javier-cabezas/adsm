#ifndef GMAC_HAL_EVENT_IMPL_H_
#define GMAC_HAL_EVENT_IMPL_H_

namespace __impl { namespace hal { namespace detail {

inline
_event::_event(bool async, type t, aspace &as) :
    gmac::util::lock_rw<_event>("_event"),
    aspace_(as),
    async_(async),
    synced_(async? false: true),
    type_(t),
    state_(None),
    err_(gmacSuccess)
{
}

inline
aspace &
_event::get_aspace()
{
    return aspace_;
}

inline
_event::type
_event::get_type() const
{
    return type_;
}

#if 0
template <typename I>
inline
typename _event::state
_event::get_state()
{
    return state_;
}
#endif

inline
hal::time_t
_event::get_time_queued() const
{
    return timeQueued_;
}


inline
hal::time_t
_event::get_time_submit() const
{
    return timeSubmit_;
}

inline
hal::time_t
_event::get_time_start() const
{
    return timeStart_;
}

inline
hal::time_t
_event::get_time_end() const
{
    return timeEnd_;
}

#if 0
template <typename I>
inline
void
_event::set_state(state s)
{
    state_ = s;
}

template <typename I>
inline
void
_event::set_synced(bool synced)
{
    synced_ = synced;
}
#endif

inline
bool
_event::is_synced() const
{
    return synced_;
}

} // namespace detail

}}

#endif /* GMAC_HAL_TYPES_IMPL_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
