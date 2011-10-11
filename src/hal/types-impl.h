#ifndef GMAC_HAL_TYPES_IMPL_H_
#define GMAC_HAL_TYPES_IMPL_H_

namespace __impl { namespace hal {

namespace detail {

template <typename D, typename B>
inline
aspace_t<D, B>::aspace_t(typename B::Context context, D &device) :
    context_(context),
    device_(device)
{
}

template <typename D, typename B>
inline
D &
aspace_t<D, B>::getDevice()
{
    return device_;
}

template <typename D, typename B>
typename B::Context &
aspace_t<D, B>::operator()()
{
    return context_;
}

template <typename D, typename B>
const typename B::Context &
aspace_t<D, B>::operator()() const
{
    return context_;
}

template <typename D, typename B>
inline
stream_t<D, B>::stream_t(typename B::Stream stream, aspace_parent_t &aspace) :
    stream_(stream),
    aspace_(aspace)
{
}

template <typename D, typename B>
inline
typename stream_t<D, B>::aspace_parent_t &
stream_t<D, B>::getPASpace()
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
event_t<D, B>::getStream()
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
typename event_t<D, B>::time_t
event_t<D, B>::getStartTime() const
{
    return start_;
}

template <typename D, typename B>
inline
typename event_t<D, B>::time_t
event_t<D, B>::getEndTime() const
{
    return end_;
}

template <typename D, typename B>
inline
typename event_t<D, B>::time_t
event_t<D, B>::getElapsedTime() const
{
    return end_ - start_;
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
