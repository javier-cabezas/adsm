#ifndef __API_CUDADRV_CONTEXT_IPP_
#define __API_CUDADRV_CONTEXT_IPP_

#include "Kernel.h"

namespace gmac { namespace cuda {

inline
void Context::call(dim3 Dg, dim3 Db, size_t shared, cudaStream_t tokens)
{
    _call = KernelConfig(Dg, Db, shared, tokens);
    _call.stream(_streamLaunch);
}

inline
void Context::argument(const void *arg, size_t size, off_t offset)
{
    _call.pushArgument(arg, size, offset);
}


inline const Stream
Context::eventStream() const
{
    return _streamLaunch;
}

}}

#endif
