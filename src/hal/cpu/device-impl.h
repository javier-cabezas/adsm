#ifndef GMAC_HAL_DEVICE_IMPL_H_
#define GMAC_HAL_DEVICE_IMPL_H_

namespace __impl { namespace hal {

namespace detail {

inline
device::device(type t, platform &platform,
                       coherence_domain &coherenceDomain) :
    platform_(platform),
    coherenceDomain_(coherenceDomain),
    type_(t)
{
}

inline
coherence_domain &
device::get_coherence_domain()
{
    return coherenceDomain_;
}

inline
const coherence_domain &
device::get_coherence_domain() const
{
    return coherenceDomain_;
}

inline
bool
device::is_integrated() const
{
    return integrated_;
}

inline
device::type
device::get_type() const
{
    return type_;
}

}

}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
