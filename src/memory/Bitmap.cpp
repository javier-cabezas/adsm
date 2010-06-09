#include "Bitmap.h"

#include <kernel/Context.h>

#include <cstring>

namespace gmac { namespace memory { namespace vm {

Bitmap::Bitmap(unsigned bits) :
    _device(NULL)
{
#ifdef BITMAP_WORD
    _entryShift = int(log2(paramPageSize));
    _size = (1 << (bits - _entryShift)) * sizeof(uint32_t);
    _bitmap = new uint32_t[_size / sizeof(uint32_t)];
#elif BITMAP_BYTE
    _entryShift = int(log2(paramPageSize));
    _size = (1 << (bits - _entryShift)) * sizeof(uint8_t);
    _bitmap = new uint8_t[_size];
#elif BITMAP_BIT
    _entryShift = int(log2(paramPageSize) + 5);;
    _pageShift = int(log2(paramPageSize));
    _bitMask = (1 << (_entryShift - _pageShift)) - 1;
    _size = (1 << (bits - _entryShift)) / 8;
    _bitmap = new uint32_t[_size / sizeof(uint32_t)];
#else
#error "Bitmap granularity not defined"
#endif
    memset(_bitmap, 0, size());
}

Bitmap::~Bitmap()
{
    delete[] _bitmap;
    Context *ctx = Context::current();
    if(_device != NULL) ctx->free(_device);
}

void Bitmap::allocate()
{
    Context *ctx = Context::current();
    if(_device == NULL) {
        trace("Allocating dirty bitmap (%zu bytes)", size());
        ctx->malloc((void **)&_device, size());
    }
}

}}}
