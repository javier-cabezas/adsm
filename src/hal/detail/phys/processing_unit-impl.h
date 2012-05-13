#ifndef GMAC_HAL_PHYS_PROCESSING_UNIT_IMPL_H_
#define GMAC_HAL_PHYS_PROCESSING_UNIT_IMPL_H_

namespace __impl { namespace hal { namespace detail { namespace phys {

inline
aspace &
processing_unit::get_paspace()
{
    return as_;
}

inline
const aspace &
processing_unit::get_paspace() const
{
    return as_;
}

inline
bool
processing_unit::is_integrated() const
{
    return integrated_;
}

inline
processing_unit::type
processing_unit::get_type() const
{
    return type_;
}

inline
bool
processing_unit::has_access(const memory &mem, memory_connection &connection)
{
    set_memory_connection::iterator it = std::find_if(connections_.begin(), connections_.end(),
                                                      [&mem](const memory_connection &conn)
                                                      {
                                                          return &mem == conn.mem;
                                                      });

    if (it != connections_.end()) {
        connection = *it;
        return true;
    }

    return false;
}

}}}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
