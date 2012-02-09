#ifndef GMAC_CORE_HPE_ADDRESS_SPACE_IMPL_H_
#define GMAC_CORE_HPE_ADDRESS_SPACE_IMPL_H_

namespace __impl { namespace core { namespace hpe {

inline
hal::aspace &
address_space::get_hal_context()
{
    return ctx_;
}

inline
const hal::aspace &
address_space::get_hal_context() const
{
    return ctx_;
}

inline
bool
address_space::has_direct_copy(memory::address_space_const_ptr _aspace) const
{
    address_space_const_ptr aspace = util::static_pointer_cast<const address_space>(_aspace);
    return ctx_.get_device().has_direct_copy(aspace->ctx_.get_device());
}

}}}

#endif /* ADDRESS_SPACE_IMPL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
