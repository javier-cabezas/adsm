#include "IOBuffer.h"
#include "Mode.h"

#include <cuda.h>

namespace gmac { namespace gpu {

IOBuffer::IOBuffer(size_t size) :
    gmac::IOBuffer(size),
    pin(false)
{
#if CUDART_VERSION >= 2020
    CUresult ret = cuMemHostAlloc(&__addr, size, CU_MEMHOSTALLOC_PORTABLE);
#else
    CUresult ret = cuMemAllocHost(__addr, size);
#endif
    if(ret == CUDA_SUCCESS) pin = true;
    else {
        __addr = malloc(size);
    }
}

IOBuffer::~IOBuffer()
{
    if(__addr == NULL) return;
    if(pin) cuMemFreeHost(__addr);
    else free(__addr);
    __addr = NULL;
}


}}

