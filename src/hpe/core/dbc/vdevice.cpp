#ifdef USE_DBC

#include "core/hpe/vdevice.h"

namespace __dbc { namespace core { namespace hpe {

void
vdevice::cleanUpContexts()
{
    parent::cleanUpContexts();
}

gmacError_t
vdevice::cleanUp()
{
    gmacError_t ret;
    ret = parent::cleanUp();
    return ret;
}

vdevice::~vdevice()
{
}

vdevice::vdevice(process_impl &proc, AddressSpaceImpl &aSpace) :
    parent(proc, aSpace)
{
}


gmacError_t
vdevice::map(accptr_t &dst, host_ptr src, size_t size, unsigned align)
{
    REQUIRES(size > 0);

    gmacError_t ret;
    ret = parent::map(dst, src, size, align);

    return ret;
}

gmacError_t
vdevice::unmap(host_ptr addr, size_t size)
{
    REQUIRES(addr != NULL);

    gmacError_t ret;
    ret = parent::unmap(addr, size);

    return ret;
}

gmacError_t
vdevice::copyToAccelerator(accptr_t acc, host_const_ptr host, size_t size)
{
    REQUIRES(acc != accptr_t(0));
    REQUIRES(host != NULL);
    REQUIRES(size > 0);

    gmacError_t ret;
    ret = parent::copyToAccelerator(acc, host, size);

    return ret;
}

gmacError_t
vdevice::copyToHost(host_ptr host, const accptr_t acc, size_t size)
{
    REQUIRES(host != NULL);
    REQUIRES(acc != accptr_t(0));
    REQUIRES(size > 0);

    gmacError_t ret;
    ret = parent::copyToHost(host, acc, size);

    return ret;
}

gmacError_t
vdevice::copyAccelerator(accptr_t dst, const accptr_t src, size_t size)
{
    REQUIRES(dst != accptr_t(0));
    REQUIRES(src != accptr_t(0));
    REQUIRES(size > 0);

    gmacError_t ret;
    ret = parent::copyAccelerator(dst, src, size);

    return ret;
}

gmacError_t
vdevice::memset(accptr_t addr, int c, size_t size)
{
    REQUIRES(addr != accptr_t(0));
    REQUIRES(size > 0);

    gmacError_t ret;
    ret = parent::memset(addr, c, size);

    return ret;
}

gmacError_t
vdevice::bufferToAccelerator(accptr_t dst, IOBufferImpl &buffer, size_t size, size_t off)
{
    REQUIRES(size > 0);
    REQUIRES(off + size <= buffer.size());

    gmacError_t ret;

    ret = parent::bufferToAccelerator(dst, buffer, size, off);

    return ret;
}

gmacError_t
vdevice::acceleratorToBuffer(IOBufferImpl &buffer, const accptr_t dst, size_t size, size_t off)
{
    REQUIRES(size > 0);
    REQUIRES(off + size <= buffer.size());

    gmacError_t ret;

    ret = parent::acceleratorToBuffer(buffer, dst, size, off);

    return ret;
}

#if 0
void
vdevice::registerKernel(gmac_kernel_id_t k, KernelImpl &kernel)
{
    REQUIRES(kernels_.find(k) == kernels_.end());

    parent::registerKernel(k, kernel);

    ENSURES(kernels_.find(k) != kernels_.end());
}

std::string
vdevice::getKernelName(gmac_kernel_id_t k) const
{
    REQUIRES(kernels_.find(k) != kernels_.end());

    std::string ret = parent::getKernelName(k);

    return ret;
}

gmacError_t 
vdevice::moveTo(__impl::core::hpe::Accelerator &acc)
{
    REQUIRES(&acc != acc_);

    gmacError_t ret;
    ret = parent::moveTo(acc);

    return ret;
}
#endif

}}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
