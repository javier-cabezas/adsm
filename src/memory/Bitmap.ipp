#ifndef __MEMORY_BITMAP_IPP
#define __MEMORY_BITMAP_IPP

namespace gmac { namespace memory { namespace vm {

#define to32bit(a) ((unsigned long)a & 0xffffffff)

inline
bool Bitmap::checkAndClear(const void *addr)
{
    bool ret = false;
#ifdef BITMAP_BIT
    size_t entry = to32bit(addr) >> (_shiftEntry + 5);
    trace("Bitmap check for %p -> entry %zu", addr, entry);
    trace("Bitmap entry before: 0x%x", _bitmap[entry]);
    uint32_t val = 1 << ((to32bit(addr) >> _shiftEntry) & _bitMask);
    if((_bitmap[entry] & val) != 0) ret = true;
    _bitmap[entry] &= ~val;
#else
    size_t entry = to32bit(addr) >> _shiftEntry;
    trace("Bitmap entry before: 0x%x", _bitmap[entry]);
    trace("Bitmap check for %p -> entry %zu", addr, entry);
#ifdef BITMAP_BYTE
    typedef uint8_t T;
#else
#ifdef BITMAP_WORD
    typedef uint32_t T;
#else
#error "Bitmap granularity not defined"
#endif
#endif
    if (_shiftEntry != _shiftPage) {
        uint32_t chunkIdx = entry & _bitMask;
#if 0
        printf("Addr:     %p\n", addr);
        printf("ChunkIdx: %d\n", chunkIdx);
#endif
        entry += chunkIdx;
    }
#if 0
    printf("Page:     %zd\n", page);
    printf("Entry:    %zd\n", entry);

    printf("Before: %x\n", _bitmap[entry]);
#endif
    if(_bitmap[entry] != 0) ret = true;
    _bitmap[entry] = 0;
#if 0
    printf("After: %x\n", _bitmap[entry]);
#endif
#endif
    trace("Bitmap entry after: 0x%x", _bitmap[entry]);
    return ret;
}

inline
void *Bitmap::device() 
{
    if (_device == NULL) allocate();
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

inline
const size_t Bitmap::shiftPage() const
{
    return _shiftPage;
}

inline
const size_t Bitmap::shiftEntry() const
{
    return _shiftEntry;
}


}}}

#endif
