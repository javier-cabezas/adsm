#ifndef __MEMORY_BLOCK_IPP
#define __MEMORY_BLOCK_IPP

#include "core/Mode.h"

namespace gmac { namespace memory {

inline Block::Block(void *addr, size_t size) :
    util::Lock("memory::Block"),
    addr_(addr),
    size_(size)
{}

inline uint8_t *
Block::addr() const
{
    return (uint8_t *) addr_;
}

inline uint8_t *
Block::end() const
{
    return addr() + size_;
}

inline size_t
Block::size() const
{
    return size_;
}

inline void
Block::lock() const
{
    return util::Lock::lock();
}

inline void
Block::unlock() const
{
    return util::Lock::unlock();
}

inline AcceleratorBlock::AcceleratorBlock(Mode &owner, void *addr, size_t size) :
    Block(addr, size),
    owner_(owner)
{ }

inline AcceleratorBlock::~AcceleratorBlock()
{ }

inline
AcceleratorBlock &
AcceleratorBlock::operator =(const AcceleratorBlock &)
{
    Fatal("Assigment of accelerator blocks is not supported");
    return *this;
}

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
