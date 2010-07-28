#ifndef __MEMORY_BLOCK_IPP
#define __MEMORY_BLOCK_IPP


namespace gmac { namespace memory {
Block::Block(void *__addr, size_t size) :
    RWLock(paraver::LockBlock),
    __addr(__addr),
    __size(__size)
{}

AcceleratorBlock::AcceleratorBlock(Context *__owner, void *__addr, size_t __size) :
    Block(__addr, __size),
    __owner(__owner)
{ }

AcceleratorBlock::~AcceleratorBlock()
{ }

SystemBlock::SystemBlock(size_t size) :
    Block(NULL, 0)
{ }

SystemBlock::~SystemBlock()
{ }

}}

#endif
