#ifndef GMAC_CORE_KERNEL_IPP_
#define GMAC_CORE_KERNEL_IPP_

namespace gmac { namespace core {

inline
Argument::Argument(void * ptr, size_t size, unsigned offset) :
    ptr_(ptr), size_(size), offset_(offset)
{
}

inline
KernelConfig::KernelConfig() :
    argsSize_(0)
{
}

inline
KernelConfig::~KernelConfig()
{
    clear();
}


inline unsigned
KernelConfig::argsSize() const
{
    return argsSize_;
}

inline uint8_t *
KernelConfig::argsArray()
{
    return stack_;
}

inline
Kernel::Kernel(const KernelDescriptor & k) :
    KernelDescriptor(k.name(), k.key())
{
}

inline
Kernel::~Kernel()
{
}


inline
KernelLaunch::~KernelLaunch()
{
}

}}

#endif
