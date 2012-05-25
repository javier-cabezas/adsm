#ifndef GMAC_HAL_EVENT_IMPL_H_
#define GMAC_HAL_EVENT_IMPL_H_

namespace __impl { namespace hal { namespace detail {

#ifdef USE_TRACE

inline
hal::time_t
operation::get_time_queued() const
{
    return timeQueued_;
}

inline
hal::time_t
operation::get_time_submit() const
{
    return timeSubmit_;
}

inline
hal::time_t
operation::get_time_start() const
{
    return timeStart_;
}

inline
hal::time_t
operation::get_time_end() const
{
    return timeEnd_;
}

#endif

inline
event::event(type t) :
    gmac::util::lock_rw<event>("event"),
    synced_(true),
    type_(t),
    state_(state::None),
    err_(HAL_SUCCESS)
{
}

inline
event::type
event::get_type() const
{
    return type_;
}

#if 0
template <typename I>
inline
typename event::state
event::get_state()
{
    return state_;
}

inline
hal::time_t
event::get_time_queued() const
{
    return timeQueued_;
}

inline
hal::time_t
event::get_time_submit() const
{
    return timeSubmit_;
}

inline
hal::time_t
event::get_time_start() const
{
    return timeStart_;
}

inline
hal::time_t
event::get_time_end() const
{
    return timeEnd_;
}

template <typename I>
inline
void
event::set_state(state s)
{
    state_ = s;
}

template <typename I>
inline
void
event::set_synced(bool synced)
{
    synced_ = synced;
}
#endif

inline
bool
event::is_synced() const
{
    return synced_;
}

template <typename Func, typename Op, typename... Args>
auto
event::queue(const Func &f, Op &op, Args... args) -> decltype(f(args...))
{
    lock_write();

#ifdef USE_TRACE
    if (operations_.size() == 0) {
        // Get the base time if this is the first operation of the event
        timeBase_ = hal::get_timestamp();
    }
#endif
    // Wait for previous operations if the new operation is not asynchronous
    if (op.is_async() == false) {
        if (syncOpBegin_ != operations_.end()) {
            (*syncOpBegin_)->sync();
        }
    }
    auto r = op.execute(f, args...);

    operations_.push_back(&op);

    // Compute the first operation to be synchronized
    if (operations_.size() == 1) {
        syncOpBegin_ = operations_.begin();
    } else if (syncOpBegin_ == operations_.end()) {
        syncOpBegin_ = std::prev(operations_.end());
    }

    unlock();

    return r;
}

template <typename Func, typename... Args>
auto
event::queue(const Func &f, cpu::operation &op, Args... args) -> decltype(f(args...))
{
    lock_write();

#ifdef USE_TRACE
    if (operations_.size() == 0) {
        // Get the base time if this is the first operation of the event
        timeBase_ = hal::get_timestamp();
    }
#endif
    // Wait for previous operations if the new operation is not asynchronous
    if (op.is_async() == false) {
        if (syncOpBegin_ != operations_.end()) {
            (*syncOpBegin_)->sync();
        }
    }
    auto r = op.execute(f, args...);

    operations_.push_back(&op);

    // Compute the first operation to be synchronized
    if (operations_.size() == 1) {
        syncOpBegin_ = operations_.begin();
    } else if (syncOpBegin_ == operations_.end()) {
        syncOpBegin_ = std::prev(operations_.end());
    }

    unlock();

    return r;
}

}}} // namespace detail

#endif /* GMAC_HAL_TYPES_IMPL_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
