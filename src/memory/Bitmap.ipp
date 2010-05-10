#ifndef __MEMORY_BITMAP_IPP
#define __MEMORY_BITMAP_IPP

#include <config/debug.h>

namespace gmac { namespace memory { namespace vm {

inline
bool Bitmap::check(const void *addr)
{
    bool ret = false;
    size_t entry = ((unsigned long)addr >> __pageShift);
    TRACE("Bitmap check for %p -> entry %zu", addr, entry);
    TRACE("Bitmap entry: 0x%x", __bitmap[entry]);
    if(__bitmap[entry] != 0) ret = true;
    __bitmap[entry] = 0;
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
    return __size;
}

}}}

#endif
