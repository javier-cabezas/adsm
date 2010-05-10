#ifndef __MEMORY_BITMAP_IPP
#define __MEMORY_BITMAP_IPP

#include <config/debug.h>

namespace gmac { namespace memory { namespace vm {

inline
bool Bitmap::check(const void *addr)
{
    bool ret = false;
    size_t entry = ((unsigned long)addr >> __pageShift);
    TRACE("Bitmap check for %p -> entry %zu bit %zu (%zu)", addr, entry >> 5, (entry & 0x1f), __pageShift);
    if((__bitmap[entry >> 5] & (1 << (entry & 0x1f))) != 0) ret = true;
    TRACE("Bitmap entry: 0x%x using mask 0x%x will return %d (new 0x%x)", __bitmap[entry >> 5], 1 << (entry & 0x1f),
        __bitmap[entry >> 5] & (1 << (entry & 0x1f)), ~(1 << (entry & 0x1f)));
    // Set to null (again)
    __bitmap[entry >> 5] &= ~(1 << (entry & 0x1f));
    return ret;
}

inline
void *Bitmap::device() 
{
    allocate();
    return __device;
}

inline
void *Bitmap::host() const
{
    return __bitmap;
}

inline
const size_t Bitmap::size() const
{
    return __size * sizeof(uint32_t);
}

}}}

#endif
