#ifndef __MEMORY_BITMAP_IPP
#define __MEMORY_BITMAP_IPP

namespace gmac { namespace memory { namespace vm {

inline
Bitmap::Bitmap(unsigned bits)
{
    size_t space = 1 << bits;
    size_t bytes = space / paramPageSize;
    if(space % paramPageSize) bytes++;
    __size = bytes / 8;
    if(bytes % 8) __size++;
    __pageShift = int(log2(paramPageSize));
    __bitmap = new uint8_t[__size];
}

inline
Bitmap::~Bitmap()
{
    delete[] __bitmap;
}

inline
void Bitmap::clear()
{
    memset(__bitmap, 0, __size * sizeof(uint8_t));
}

inline
bool Bitmap::check(const void *addr)
{
    bool ret = false;
    size_t entry = ((unsigned long)addr >> __pageShift);
    if((__bitmap[entry << 3] & (entry & 0x7)) != 0) ret = true;
    return ret;
}

}}}

#endif
