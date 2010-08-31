#ifndef __MEMORY_BLOCK_IPP
#define __MEMORY_BLOCK_IPP


namespace gmac { namespace memory {
inline Block::Block(void *addr, size_t size) :
    RWLock(paraver::LockBlock),
    __addr(addr),
    __size(size)
{}

inline AcceleratorBlock::AcceleratorBlock(Mode *owner, void *addr, size_t size) :
    Block(addr, size),
    __owner(owner)
{ }

inline AcceleratorBlock::~AcceleratorBlock()
{ }

template<typename T>
inline SystemBlock<T>::SystemBlock(void *addr, size_t size, T state) :
    Block(addr, size),
    __state(state)
{ }

template<typename T>
inline SystemBlock<T>::~SystemBlock()
{ }

template<typename T>
inline T SystemBlock<T>::state()
{ 
    lockRead();
    T ret = __state;
    unlock();
    return ret;
}

template<typename T>
inline void SystemBlock<T>::state(T s)
{
    lockWrite();
    __state = s;
    unlock();
}

}}

#endif
