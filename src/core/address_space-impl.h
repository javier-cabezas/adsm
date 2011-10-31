#ifndef GMAC_CORE_ADDRESS_SPACE_IMPL_H_
#define GMAC_CORE_ADDRESS_SPACE_IMPL_H_

namespace __impl { namespace core {

inline
address_space::address_space() :
    util::Reference("AddressSpace"),
    map_("AddressSpace")
{
}

inline
address_space::~address_space()
{
}

inline
memory::map_object &
address_space::get_object_map()
{
    return map_;
}

inline
const memory::map_object &
address_space::get_object_map() const
{
    return map_;
}

}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
