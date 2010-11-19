#ifndef GMAC_MEMORY_TEST_ACCELERATORBLOCK_IPP_
#define GMAC_MEMORY_TEST_ACCELERATORBLOCK_IPP_

namespace __dbc { namespace memory {

inline
AcceleratorBlock::AcceleratorBlock(__impl::core::Mode &owner, void *addr, size_t size) :
    __impl::memory::AcceleratorBlock(owner, addr, size)
{
    REQUIRES(size > 0);
    REQUIRES(addr != NULL);
}

inline
AcceleratorBlock::~AcceleratorBlock()
{
}

}}

#endif
