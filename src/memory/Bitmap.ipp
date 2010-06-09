#ifndef __MEMORY_BITMAP_IPP
#define __MEMORY_BITMAP_IPP

namespace gmac { namespace memory { namespace vm {

inline
bool Bitmap::checkAndClear(const void *addr)
{
    bool ret = false;
    size_t entry = ((unsigned long)addr >> _entryShift);
    trace("Bitmap check for %p -> entry %zu", addr, entry);
    trace("Bitmap entry: 0x%x", _bitmap[entry]);
#if BITMAP_BIT
    uint32_t val = 1 << (((unsigned long)addr >> _pageShift) & _bitMask);
    if(_bitmap[entry] & val) ret = true;
    _bitmap[entry] |= val;
#else
    if(_bitmap[entry] != 0) ret = true;
    _bitmap[entry] = 0;
#endif
    return ret;
}

inline
void *Bitmap::device() 
{
    allocate();
    return _device;
}

inline
void *Bitmap::host() const
{
    return _bitmap;
}

inline
const size_t Bitmap::size() const
{
    return _size;
}

}}}

#endif
