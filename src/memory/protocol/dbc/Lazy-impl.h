
#ifndef GMAC_MEMORY_PROTOCOL_DBC_LAZY_IMPL_H_
#define GMAC_MEMORY_PROTOCOL_DBC_LAZY_IMPL_H_

#include "memory/protocol/lazy/LazyTypes.h"

namespace __dbc { namespace memory { namespace protocol {

template <typename T>
Lazy<T>::Lazy(unsigned limit) :
    Parent(limit)
{
}

template <typename T>
Lazy<T>::~Lazy()
{
}

template <typename T>
gmacError_t
Lazy<T>::signalRead(BlockImpl &_block, hostptr_t addr)
{
    LazyBlockImpl &block = dynamic_cast<LazyBlockImpl &>(_block);
    //REQUIRES(block.getState() == __impl::memory::protocol::lazy::Invalid);

    gmacError_t ret = Parent::signalRead(block, addr);

    return ret;
}

template <typename T>
gmacError_t
Lazy<T>::signalWrite(BlockImpl &_block, hostptr_t addr)
{
    LazyBlockImpl &block = dynamic_cast<LazyBlockImpl &>(_block);
    gmacError_t ret = Parent::signalWrite(block, addr);

    ENSURES(block.getState() == __impl::memory::protocol::lazy::Dirty);

    return ret;
}

template <typename T>
gmacError_t
Lazy<T>::acquire(BlockImpl &_block)
{
    LazyBlockImpl &block = dynamic_cast<LazyBlockImpl &>(_block);

    ENSURES(block.getState() == __impl::memory::protocol::lazy::ReadOnly ||
            block.getState() == __impl::memory::protocol::lazy::Invalid);

    gmacError_t ret = Parent::acquire(block);
    
    ENSURES(block.getState() == __impl::memory::protocol::lazy::Invalid);

    return ret;
}

template <typename T>
gmacError_t
Lazy<T>::release(BlockImpl &_block)
{
    LazyBlockImpl &block = dynamic_cast<LazyBlockImpl &>(_block);
    gmacError_t ret = Parent::release(block);

    ENSURES(block.getState() == __impl::memory::protocol::lazy::ReadOnly);

    return ret;
}

template <typename T>
gmacError_t
Lazy<T>::releaseObjects()
{
    gmacError_t ret = Parent::releaseObjects();

    ENSURES(Parent::dbl_.size() == 0);

    return ret;
}

template <typename T>
gmacError_t
Lazy<T>::toHost(BlockImpl &_block)
{
    LazyBlockImpl &block = dynamic_cast<LazyBlockImpl &>(_block);
    gmacError_t ret = Parent::toHost(block);

    ENSURES(block.getState() != __impl::memory::protocol::lazy::Invalid);

    return ret;
}

template <typename T>
gmacError_t
Lazy<T>::copyToBuffer(BlockImpl &block, IOBufferImpl &buffer, size_t size, 
    size_t bufferOffset, size_t blockOffset)
{
    REQUIRES(blockOffset  + size <= block.size());
    REQUIRES(bufferOffset + size <= buffer.size());

    gmacError_t ret = Parent::copyToBuffer(block, buffer, size, bufferOffset, blockOffset);

    return ret;
}

template <typename T>
gmacError_t
Lazy<T>::copyFromBuffer(BlockImpl &block, IOBufferImpl &buffer, size_t size,
    size_t bufferOffset, size_t blockOffset)
{
    REQUIRES(blockOffset  + size <= block.size());
    REQUIRES(bufferOffset + size <= buffer.size());

    gmacError_t ret = Parent::copyFromBuffer(block, buffer, size, bufferOffset, blockOffset);

    return ret;
}

template <typename T>
gmacError_t
Lazy<T>::memset(const BlockImpl &block, int v, size_t size, size_t blockOffset)
{
    REQUIRES(blockOffset + size <= block.size());

    gmacError_t ret = Parent::memset(block, v, size, blockOffset);

    return ret;
}

template <typename T>
gmacError_t
Lazy<T>::flushDirty()
{
    gmacError_t ret = Parent::flushDirty();

    ENSURES(Parent::dbl_.size() == 0);

    return ret;
}

}}}

#endif //GMAC_MEMORY_PROTOCOL_DBC_LAZY_IMPL_H_

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
