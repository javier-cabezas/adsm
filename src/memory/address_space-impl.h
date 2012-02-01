#ifndef GMAC_MEMORY_ADDRESS_SPACE_IMPL_H_
#define GMAC_MEMORY_ADDRESS_SPACE_IMPL_H_

namespace __impl { namespace memory {

inline
address_space::address_space(location loc) :
    map_("AddressSpace"),
    location_(loc)
{
}

inline
address_space::~address_space()
{
    printf("Cucu\n");
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

inline
address_space::location
address_space::get_location() const
{
    return location_;
}

}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
