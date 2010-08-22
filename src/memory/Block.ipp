#ifndef __MEMORY_BLOCK_IPP
#define __MEMORY_BLOCK_IPP


namespace gmac { namespace memory {
inline Block::Block(void *addr, size_t size) :
    RWLock(paraver::LockBlock),
    __addr(__addr),
    __size(__size)
{}

inline AcceleratorBlock::AcceleratorBlock(Mode *owner, void *addr, size_t size) :
    Block(addr, size),
    __owner(owner)
{ }

inline AcceleratorBlock::~AcceleratorBlock()
{ }

template<typename T>
inline SystemBlock<T>::SystemBlock(void *addr, size_t size) :
    Block(addr, 0)
{ }

template<typename T>
inline SystemBlock<T>::~SystemBlock()
{ }

}}

#endif
