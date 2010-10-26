#ifndef __MEMORY_BLOCK_IPP
#define __MEMORY_BLOCK_IPP

#include "core/Mode.h"

namespace gmac { namespace memory {
inline Block::Block(void *addr, size_t size) :
    util::Lock("memory::Block"),
    addr_(addr),
    size_(size)
{}

inline AcceleratorBlock::AcceleratorBlock(Mode &owner, void *addr, size_t size) :
    Block(addr, size),
    owner_(owner)
{ }

inline AcceleratorBlock::~AcceleratorBlock()
{ }

#if 0
inline gmacError_t AcceleratorBlock::toDevice(off_t off, Block &block)
{
    trace("Mode %d is putting %p into device @ %p %zd bytes", owner_.id(), block.addr(), addr() + off, block.size());
    return owner_.copyToDevice(addr() + off, block.addr(), block.size());
}

inline gmacError_t AcceleratorBlock::toHost(off_t off, Block &block)
{
    trace("Mode %d is getting %p from device @ %p %zd bytes", owner_.id(), block.addr(), addr() + off, block.size());
    return owner_.copyToHost(block.addr(), addr() + off, block.size());
}

inline gmacError_t AcceleratorBlock::toHost(off_t off, void *hostAddr, size_t count)
{
    trace("Mode %d is getting %p from device @ %p %zd bytes", owner_.id(), hostAddr, addr() + off, count);
    return owner_.copyToHost(hostAddr, addr() + off, count);
}
#endif

template<typename T>
inline SystemBlock<T>::SystemBlock(void *addr, size_t size, T state) :
    Block(addr, size),
    state_(state)
{
    trace("Creating system block @ %p with %zd bytes", addr, size);
}

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
