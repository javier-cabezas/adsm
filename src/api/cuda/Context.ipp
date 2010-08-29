#ifndef __API_CUDADRV_CONTEXT_IPP_
#define __API_CUDADRV_CONTEXT_IPP_

#include "Kernel.h"

namespace gmac { namespace gpu {

inline
void Context::call(dim3 Dg, dim3 Db, size_t shared, cudaStream_t tokens)
{
    __call = KernelConfig(Dg, Db, shared, tokens);
}

inline
void Context::argument(const void *arg, size_t size, off_t offset)
{
    __call.pushArgument(arg, size, offset);
}


}}

#endif
