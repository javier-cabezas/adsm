#ifndef __MEMORY_BLOCK_IPP
#define __MEMORY_BLOCK_IPP


namespace gmac { namespace memory {
Block::Block(void *__addr, size_t size) :
    __addr(__addr),
    __size(__size)
{}

AcceleratorBlock::AcceleratorBlock(Context *__owner, size_t __size) :
    Block(NULL, 0),
    __owner(__owner)
{
    __owner->malloc(&__addr, __size);
}

AcceleratorBlock::~AcceleratorBlock()
{
    if(__addr != NULL) __owner->free(__addr);
}

SystemBlock::SystemBlock(size_t size) :
    Block(NULL, 0)
{
}

SystemBlock::~SystemBlock()
{
    if(_addr != NULL) ::free(__addr);
}

GlobalBlock::GlobalBlock(Context *__owner, size_t size) :
    Block(NULL, 0),
    __owner(__owner)
{
    __owner->mallocPageLocked(&__addr, size);
}

GlobalBlock::~GlobalBlock()
{
    if(__addr != NULL) __owner->hostFree(__addr);
}
}}

#endif
