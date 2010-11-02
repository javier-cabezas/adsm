#ifndef GMAC_MEMORY_TEST_ACCELERATORBLOCK_IPP_
#define GMAC_MEMORY_TEST_ACCELERATORBLOCK_IPP_

namespace gmac { namespace memory { namespace __dbc {

inline
AcceleratorBlock::AcceleratorBlock(Mode &owner, void *addr, size_t size) :
    __impl::AcceleratorBlock(owner, addr, size)
{
    REQUIRES(size > 0);
    REQUIRES(addr != NULL);
}

inline
AcceleratorBlock::~AcceleratorBlock()
{
}

}}}

#endif
