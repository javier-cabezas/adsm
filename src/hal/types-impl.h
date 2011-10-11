#ifndef GMAC_HAL_TYPES_IMPL_H_
#define GMAC_HAL_TYPES_IMPL_H_

#ifndef _MSC_VER
#include <sys/time.h>
#endif

namespace __impl { namespace hal {

#ifdef _MSC_VER
time_t get_timestamp()
{
    FILETIME ft;
    GetSystemTimeAsFileTime(&ft);

    unsigned __int64 tmp = 0;
    tmp |= ft.dwHighDateTime;
    tmp <<= 32;
    tmp |= ft.dwLowDateTime;
    tmp -= DELTA_EPOCH_IN_MICROSECS;
    tmp /= 10;

    return time_t(tmp);
}
#else
time_t get_timestamp()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    time_t ret;

    ret = tv.tv_sec * 1000000 + tv.tv_usec;
    return ret;
}
#endif

namespace detail {

template <typename D, typename B>
inline
aspace_t<D, B>::aspace_t(typename B::context context, D &dev) :
    context_(context),
    device_(dev)
{
}

template <typename D, typename B>
inline
D &
aspace_t<D, B>::get_device()
{
    return device_;
}

template <typename D, typename B>
typename B::context &
aspace_t<D, B>::operator()()
{
    return context_;
}

template <typename D, typename B>
const typename B::context &
aspace_t<D, B>::operator()() const
{
    return context_;
}

template <typename D, typename B>
inline
stream_t<D, B>::stream_t(typename B::stream stream, aspace_parent_t &aspace) :
    stream_(stream),
    aspace_(aspace)
{
}

template <typename D, typename B>
inline
typename stream_t<D, B>::aspace_parent_t &
stream_t<D, B>::get_address_space()
{
    return aspace_;
}

template <typename D, typename B>
inline
event_t<D, B>::event_t(stream_parent_t &stream, gmacError_t err) :
    stream_(stream),
    err_(err)
{
}

template <typename D, typename B>
inline
typename event_t<D, B>::stream_parent_t &
event_t<D, B>::get_stream()
{
    return stream_;
}

template <typename D, typename B>
inline
gmacError_t
event_t<D, B>::getError() const
{
    return err_;
}

template <typename D, typename B>
inline
hal::time_t
event_t<D, B>::get_time_queued() const
{
    return timeQueued_;
}


template <typename D, typename B>
inline
hal::time_t
event_t<D, B>::get_time_submit() const
{
    return timeSubmit_;
}

template <typename D, typename B>
inline
hal::time_t
event_t<D, B>::get_time_start() const
{
    return timeStart_;
}

template <typename D, typename B>
inline
hal::time_t
event_t<D, B>::get_time_end() const
{
    return timeEnd_;
}

template <typename D, typename B>
inline
async_event_t<D, B>::async_event_t(stream_parent_t &stream, gmacError_t err) :
    event_t<D, B>(stream, err)
{
}

template <typename D, typename B>
inline
bool
async_event_t<D, B>::isSynced() const
{
    return synced_;
}

template <typename D, typename B>
inline
void
async_event_t<D, B>::setSynced(bool synced)
{
    synced_ = synced;
}

} // namespace detail

}}

#endif /* GMAC_HAL_TYPES_IMPL_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
