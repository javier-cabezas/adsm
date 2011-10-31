#ifndef GMAC_HAL_DEVICE_IMPL_H_
#define GMAC_HAL_DEVICE_IMPL_H_

namespace __impl { namespace hal {

namespace detail {

template <typename I> 
inline
device<I>::device(typename I::coherence_domain &coherenceDomain) :
    coherenceDomain_(coherenceDomain)
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

}

}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
