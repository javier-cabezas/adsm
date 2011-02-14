#ifdef USE_DBC
#include "api/cuda/Mode.h"

namespace __dbc { namespace cuda {

Mode::Mode(__impl::core::Process &proc, __impl::cuda::Accelerator &acc) :
    __impl::cuda::Mode(proc, acc)
{
}

Mode::~Mode()
{
}

gmacError_t
Mode::bufferToAccelerator(accptr_t dst, __impl::core::IOBuffer &buffer, size_t size, size_t off)
{
    REQUIRES(buffer.size() - off >= size);
    REQUIRES(dst != NULL);
    REQUIRES(buffer.state() == __impl::core::IOBuffer::Idle);

    gmacError_t ret = __impl::cuda::Mode::bufferToAccelerator(dst, buffer, size, off);

    return ret;
}

gmacError_t
Mode::acceleratorToBuffer(__impl::core::IOBuffer &buffer, const accptr_t src, size_t size, size_t off)
{
    REQUIRES(buffer.size() - off >= size);
    REQUIRES(src != NULL);
    REQUIRES(buffer.state() == __impl::core::IOBuffer::Idle);

    gmacError_t ret = __impl::cuda::Mode::acceleratorToBuffer(buffer, src, size, off);

    return ret;
}


}}
#endif
/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
