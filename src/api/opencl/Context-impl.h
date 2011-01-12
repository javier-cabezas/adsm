#ifndef GMAC_API_CUDA_CONTEXT_IMPL_H_
#define GMAC_API_CUDA_CONTEXT_IMPL_H_

#include "Accelerator.h"
#include "Kernel.h"

namespace __impl { namespace opencl {

inline void
Context::call(cl_uint work_dim, size_t *global_work_offset, size_t *global_work_size, size_t *local_work_size)
{
    call_ = KernelConfig(work_dim, global_work_offset, global_work_size, local_work_size, streamLaunch_);
}

inline
void Context::argument(const void *arg, size_t size)
{
    call_.pushArgument(arg, size);
}

inline const cl_command_queue
Context::eventStream() const
{
    return streamLaunch_;
}

inline Accelerator &
Context::accelerator()
{
    return dynamic_cast<Accelerator &>(acc_);
}

}}

#endif
