#ifndef GMAC_API_OPENCL_HPE_CONTEXT_IMPL_H_
#define GMAC_API_OPENCL_HPE_CONTEXT_IMPL_H_

#include "Kernel.h"

#include "api/opencl/hpe/Accelerator.h"

namespace __impl { namespace opencl { namespace hpe {


#if 0
inline gmacError_t
Context::argument(const void *arg, size_t size, unsigned index)
{
    call_.setArgument(arg, size, index);
    // TODO: perform some checking
    return gmacSuccess;
}
#endif

inline const cl_command_queue
Context::eventStream() const
{
    return stream_;
}

}}}

#endif
