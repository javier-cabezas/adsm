#ifndef GMAC_API_CUDA_CONTEXT_IPP_
#define GMAC_API_CUDA_CONTEXT_IPP_

#include "Accelerator.h"
#include "Kernel.h"

namespace gmac { namespace cuda {

inline void
Context::call(dim3 Dg, dim3 Db, size_t shared, cudaStream_t tokens)
{
    call_ = KernelConfig(Dg, Db, shared, tokens);
    call_.stream(streamLaunch_);
}

inline
void Context::argument(const void *arg, size_t size, off_t offset)
{
    call_.pushArgument(arg, size, offset);
}

inline const CUstream
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
