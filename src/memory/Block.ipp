#ifndef __MEMORY_BLOCK_IPP
#define __MEMORY_BLOCK_IPP

#include "core/Mode.h"

namespace gmac { namespace memory {
inline Block::Block(void *addr, size_t size) :
    Lock("memory::Block"),
    addr_(addr),
    size_(size)
{}

inline AcceleratorBlock::AcceleratorBlock(Mode &owner, void *addr, size_t size) :
    Block(addr, size),
    owner_(owner)
{ }

inline AcceleratorBlock::~AcceleratorBlock()
{ }

inline gmacError_t AcceleratorBlock::toDevice(off_t off, Block &block)
{
    trace("Mode %d is putting %p into device @ %p", owner_.id(), block.addr(), (uint8_t *)addr_ + off);
    return owner_.copyToDevice((uint8_t *)addr_ + off, block.addr(), block.size());
}

inline gmacError_t AcceleratorBlock::toHost(off_t off, Block &block)
{
    trace("Mode %d is getting %p from device @ %p", owner_.id(), block.addr(), (uint8_t *)addr_ + off);
    return owner_.copyToHost(block.addr(), (uint8_t *)addr_ + off, block.size());
}

inline gmacError_t AcceleratorBlock::toHost(off_t off, void *hostAddr, size_t count)
{
    trace("Mode %d is getting %p from device @ %p", owner_.id(), hostAddr, (uint8_t *)addr_ + off);
    return owner_.copyToHost(hostAddr, (uint8_t *)addr_ + off, count);
}

template<typename T>
inline SystemBlock<T>::SystemBlock(void *addr, size_t size, T state) :
    Block(addr, size),
    state_(state)
{ }

template<typename T>
inline SystemBlock<T>::~SystemBlock()
{ }


template<typename T>
inline T SystemBlock<T>::state() const
{ 
    T ret = state_;
    return ret;
}

template<typename T>
inline void SystemBlock<T>::state(T s)
{
    state_ = s;
}

}}

#endif
