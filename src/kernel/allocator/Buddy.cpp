#include "Buddy.h"

#include <util/Logger.h>

namespace gmac { namespace kernel { namespace allocator {

Buddy::Buddy(size_t size) :
    _size(round(size)),
    _index(index(_index))
{
    initMemory();
}

Buddy::~Buddy()
{
    finiMemory();
}

uint8_t Buddy::ones(register uint32_t x) const
{
    /* 32-bit recursive reduction using SWAR...
       but first step is mapping 2-bit values
       into sum of 2 1-bit values in sneaky way
    */
    x -= ((x >> 1) & 0x55555555);
    x = (((x >> 2) & 0x33333333) + (x & 0x33333333));
    x = (((x >> 4) + x) & 0x0f0f0f0f);
    x += (x >> 8);
    x += (x >> 16);
    return(x & 0x0000003f);
}

uint8_t Buddy::index(register uint32_t x) const
{
    register int32_t y = (x & (x - 1));

    y |= -y;
    y >>= 31;
    x |= (x >> 1);
    x |= (x >> 2);
    x |= (x >> 4);
    x |= (x >> 8);
    x |= (x >> 16);
    return(ones(x >> 1) - y);
}

uint32_t Buddy::round(register uint32_t x) const
{
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x++;

    return x;
}

void *Buddy::getFromList(uint8_t i)
{
    if(i > _index) return NULL;
    /* Check for spare chunks of the requested size */
    List &list = _tree[i];
    if(list.empty() == false) {
        void * ret = list.front();
        list.pop_front();
        return ret;
    }

    /* No spare chunks, try splitting a bigger one */
    void *larger = getFromList(i + 1);
    if(larger == NULL) return NULL; /* Not enough memory */
    void *mid = (uint8_t *)larger + (1 << i);
    list.push_back(mid);
    return larger;
}

void Buddy::putToList(void *addr, uint8_t i)
{
    if(i == _index) {
        _tree[i].push_back(addr);
        return;
    }
    
    /* Try merging buddies */
    unsigned long mask = ~((1 << (i + 1)) - 1);
    List &list = _tree[i];
    List::iterator buddy;
    for(buddy = list.begin(); buddy != list.end(); buddy++) {
        if(((unsigned long)*buddy & mask) != ((unsigned long )addr & mask))
            continue;
        return putToList((void *)((unsigned long)addr & mask), i + 1);        
    }
    return;

}

void *Buddy::get(size_t size)
{
    uint8_t i = index(size);
    return getFromList(i);
}

void Buddy::put(void *addr, size_t size)
{
    uint8_t i = index(size);
    return putToList(addr, i);
}

}}}
