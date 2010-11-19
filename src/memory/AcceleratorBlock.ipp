#ifndef GMAC_MEMORY_ACCELERATORBLOCK_IPP_
#define GMAC_MEMORY_ACCELERATORBLOCK_IPP_

namespace __impl { namespace memory { 

inline AcceleratorBlock::AcceleratorBlock(core::Mode &owner, void *addr, size_t size) :
    Block(addr, size),
    owner_(owner)
{
}

inline AcceleratorBlock::~AcceleratorBlock()
{
}

inline
AcceleratorBlock &
AcceleratorBlock::operator =(const AcceleratorBlock &)
{
    FATAL("Assigment of accelerator blocks is not supported");
    return *this;
}


inline
core::Mode &
AcceleratorBlock::owner()
{
    return owner_;
}


}}

#endif
