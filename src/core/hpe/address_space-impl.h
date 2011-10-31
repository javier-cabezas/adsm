#ifndef GMAC_CORE_HPE_ADDRESS_SPACE_IMPL_H_
#define GMAC_CORE_HPE_ADDRESS_SPACE_IMPL_H_

namespace __impl { namespace core { namespace hpe {

inline
hal::context_t &
address_space::get_hal_context()
{
    return ctx_;
}

inline
const hal::context_t &
address_space::get_hal_context() const
{
    return ctx_;
}

inline
bool
address_space::has_direct_copy(const core::address_space &_aspace) const
{
    const address_space &aspace = reinterpret_cast<const address_space &>(_aspace);
    return ctx_.get_device().has_direct_copy(aspace.ctx_.get_device());
}

}}}

#endif /* ADDRESS_SPACE_IMPL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
