#ifndef GMAC_API_CUDA_CONTEXT_IPP_
#define GMAC_API_CUDA_CONTEXT_IPP_

#include "Kernel.h"

namespace gmac { namespace cuda {

inline
Context &
Context::current()
{
    return dynamic_cast<Context &>(gmac::Context::current());
}

inline void
Context::call(dim3 Dg, dim3 Db, size_t shared, cudaStream_t tokens)
{
    _call = KernelConfig(Dg, Db, shared, tokens);
    _call.stream(_streamLaunch);
}

inline
void Context::argument(const void *arg, size_t size, off_t offset)
{
    _call.pushArgument(arg, size, offset);
}

inline const CUstream
Context::eventStream() const
{
    return _streamLaunch;
}

inline Accelerator &
Context::accelerator()
{
    return static_cast<Accelerator &>(acc_);
}

}}

#endif
