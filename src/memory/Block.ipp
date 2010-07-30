#ifndef __MEMORY_BLOCK_IPP
#define __MEMORY_BLOCK_IPP


namespace gmac { namespace memory {
Block::Block(void *addr, size_t size) :
    RWLock(paraver::LockBlock),
    __addr(__addr),
    __size(__size)
{}

AcceleratorBlock::AcceleratorBlock(Mode *owner, void *addr, size_t size) :
    Block(addr, size),
    __owner(owner)
{ }

AcceleratorBlock::~AcceleratorBlock()
{ }

template<typename T>
SystemBlock<T>::SystemBlock(void *addr, size_t size) :
    Block(addr, 0)
{ }

template<typename T>
SystemBlock<T>::~SystemBlock()
{ }

}}

#endif
