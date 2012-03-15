#ifndef GMAC_HAL_PHYS_PROCESSING_UNIT_IMPL_H_
#define GMAC_HAL_PHYS_PROCESSING_UNIT_IMPL_H_

namespace __impl { namespace hal { namespace detail { namespace phys {

inline
processing_unit::set_aspace &
processing_unit::get_paspaces()
{
    return aspaces_;
}

inline
const processing_unit::set_aspace &
processing_unit::get_paspaces() const
{
    return aspaces_;
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
processing_unit::has_access(memory_ptr mem, memory_connection &connection)
{
    set_memory_connection::iterator it = std::find_if(memories_.begin(), memories_.end(),
                                                      [&](const memory_connection &conn)
                                                      {
                                                          return mem->get_id() == conn.mem->get_id();
                                                      });

    if (it != memories_.end()) {
        connection = *it;
        return true;
    }

    return false;
}

}

}}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
