#ifndef GMAC_HAL_DEVICE_IMPL_H_
#define GMAC_HAL_DEVICE_IMPL_H_

namespace __impl { namespace hal {

namespace detail {

template <typename I> 
void
platform<I>::add_device(typename I::device &d)
{
    devices_.push_back(&d);
}

template <typename I> 
unsigned
platform<I>::get_ndevices()
{
    return unsigned(devices_.size());
}

template <typename I> 
typename platform<I>::list_device
platform<I>::get_devices()
{
    return devices_;
}

template <typename I> 
inline
device<I>::device(type t, typename I::coherence_domain &coherenceDomain) :
    coherenceDomain_(coherenceDomain),
    type_(t)
{
}

template <typename I> 
inline
typename I::coherence_domain &
device<I>::get_coherence_domain()
{
    return coherenceDomain_;
}

template <typename I> 
inline
const typename I::coherence_domain &
device<I>::get_coherence_domain() const
{
    return coherenceDomain_;
}

template <typename I> 
inline
bool
device<I>::is_integrated() const
{
    return integrated_;
}

template <typename I> 
inline
typename device<I>::type
device<I>::get_type() const
{
    return type_;
}

}

}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
