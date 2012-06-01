#ifndef GMAC_HAL_DETAIL_VIRT_SCHEDULER_H_
#define GMAC_HAL_DETAIL_VIRT_SCHEDULER_H_

namespace __impl { namespace hal { namespace detail { namespace virt {

class context;

class GMAC_LOCAL scheduler
{
public:
    virtual hal::error add_context(context &ctx) = 0;
};

}}}}

#endif /* ASPACE_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
