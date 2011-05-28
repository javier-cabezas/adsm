#ifdef USE_DBC
#include "api/opencl/hpe/Accelerator.h"
#include "api/opencl/IOBuffer.h"

#include "core/hpe/Mode.h"

namespace __dbc { namespace opencl { namespace hpe {

Accelerator::Accelerator(int n, cl_context context, cl_device_id device) :
    __impl::opencl::hpe::Accelerator(n, context, device)
{
    REQUIRES(n >= 0);
}

Accelerator::~Accelerator()
{
}

gmacError_t Accelerator::copyToAccelerator(accptr_t acc, const hostptr_t host, size_t size, __impl::core::hpe::Mode &mode)
{
    // PRECONDITIONS
    REQUIRES(acc  != 0);
    REQUIRES(host != NULL);
    REQUIRES(size > 0);
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::opencl::hpe::Accelerator::copyToAccelerator(acc, host, size, mode);
    // POSTCONDITIONS
    ENSURES(ret == gmacSuccess);

    return ret;
}

gmacError_t Accelerator::copyToHost(hostptr_t host, const accptr_t acc, size_t size, __impl::core::hpe::Mode &mode)
{
    // PRECONDITIONS
    REQUIRES(host != NULL);
    REQUIRES(acc  != 0);
    REQUIRES(size > 0);
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::opencl::hpe::Accelerator::copyToHost(host, acc, size, mode);
    // POSTCONDITIONS
    ENSURES(ret == gmacSuccess);

    return ret;
}

gmacError_t Accelerator::copyAccelerator(accptr_t dst, const accptr_t src, size_t size)
{
    // PRECONDITIONS
    REQUIRES(src != 0);
    REQUIRES(dst != 0);
    REQUIRES(size > 0);
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::opencl::hpe::Accelerator::copyAccelerator(dst, src, size);
    // POSTCONDITIONS
    ENSURES(ret == gmacSuccess);

    return ret;
}

#if 0
gmacError_t Accelerator::copyToAcceleratorAsync(accptr_t acc, __impl::opencl::IOBuffer &buffer, size_t bufferOff, size_t count, __impl::core::hpe::Mode &mode, cl_command_queue stream)
{
    // PRECONDITIONS
    REQUIRES(count > 0);
    REQUIRES(acc != 0);
    REQUIRES(buffer.addr() != NULL);
    REQUIRES(buffer.size() > 0);
    REQUIRES(bufferOff + count <= buffer.size());
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::opencl::hpe::Accelerator::copyToAcceleratorAsync(acc, buffer, bufferOff, count, mode, stream);
    // POSTCONDITIONS
    ENSURES(ret == gmacSuccess);

    return ret;
}

gmacError_t Accelerator::copyToHostAsync(__impl::opencl::IOBuffer &buffer, size_t bufferOff, const accptr_t acc, size_t count, __impl::core::hpe::Mode &mode, cl_command_queue stream)
{
    // PRECONDITIONS
    REQUIRES(count > 0);
    REQUIRES(acc != 0);
    REQUIRES(buffer.addr() != NULL);
    REQUIRES(buffer.size() > 0);
    REQUIRES(bufferOff + count <= buffer.size());
    // CALL IMPLEMENTATION
    gmacError_t ret = __impl::opencl::hpe::Accelerator::copyToHostAsync(buffer, bufferOff, acc, count, mode, stream);
    // POSTCONDITIONS
    ENSURES(ret == gmacSuccess);

    return ret;
}
#endif

}}}
#endif
/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
