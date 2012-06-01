#ifndef GMAC_HAL_DETAIL_VIRT_CONTEXT_IMPL_H_
#define GMAC_HAL_DETAIL_VIRT_CONTEXT_IMPL_H_

namespace __impl { namespace hal { namespace detail { namespace virt {

inline
context::context(aspace *as, priority prio) :
    lock("context"),
    aspace_(as),
    prio_(prio)
{
}

inline
aspace *
context::get_aspace()
{
    return aspace_;
}

inline
const aspace *
context::get_aspace() const
{
    return aspace_;
}

inline
void
context::set_priority(priority prio)
{
    prio_ = prio;
}

inline
priority
context::get_priority() const
{
    return prio_;
}

}}}}

#endif /* STREAM_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
