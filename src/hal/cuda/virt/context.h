#ifndef GMAC_HAL_CUDA_VIRT_CONTEXT_H_
#define GMAC_HAL_CUDA_VIRT_CONTEXT_H_

#include "hal/detail/types.h"
#include "hal/detail/virt/context.h"

namespace __impl { namespace hal { namespace cuda { namespace virt {

typedef hal::detail::code::kernel hal_kernel;

typedef hal::detail::virt::context hal_context;
typedef hal::detail::virt::priority hal_priority;

class GMAC_LOCAL context :
    public hal_context {

protected:
    stream *stream_;

public:
    context(virt::aspace *as, hal_priority prio, hal::error &err);
    virtual ~context();

    hal::error migrate(hal_aspace &as)
    {
        FATAL("Migration not implemented in CUDA");
        return hal::error::HAL_SUCCESS;
    }

    context::state get_state()
    {
        return context::state::Empty;
    }

    hal_event_ptr queue(const hal_kernel &_k, hal_kernel::config &_config, hal_kernel::arg_list &_args, hal::error &err);
};

}}}}

#endif /* STREAM_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
