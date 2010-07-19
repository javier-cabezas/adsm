#include "Bitmap.h"

#include <kernel/Context.h>

#include <cstring>

namespace gmac { namespace memory { namespace vm {

Bitmap::Bitmap(unsigned bits) :
    _device(NULL)
{
    _shiftPage = int(log2(paramPageSize));
    _shiftEntry = int(log2(paramPageSize / paramBitmapChunksPerPage));
#ifdef BITMAP_BIT
    _bitMask = (1 << 5) - 1;
    _size = (1 << (bits - _shiftEntry)) / 8;
    _bitmap = new uint32_t[_size / sizeof(uint32_t)];
#else
    _bitMask = (1 << (_shiftPage - _shiftEntry)) - 1;
#ifdef BITMAP_BYTE
    typedef uint8_t T;
#else
#ifdef BITMAP_WORD
    typedef uint32_t T;
#else
#error "Bitmap granularity not defined"
#endif
#endif
    _size = (1 << (bits - _shiftEntry)) * sizeof(T);
#if 0
    _bitmap = new T[_size / sizeof(T)];
#endif
#endif
}

Bitmap::~Bitmap()
{
    //delete[] _bitmap;
    Context *ctx = Context::current();
    if(_device != NULL) ctx->hostFree(_bitmap);
}

void Bitmap::allocate()
{
    assertion(_device == NULL);
    Context *ctx = Context::current();
    trace("Allocating dirty bitmap (%zu bytes)", size());
    ctx->mallocPageLocked((void **)&_bitmap, _size);
    memset(_bitmap, 0, size());
#if 0
    ctx->malloc((void **)&_device, size());
#endif
    ctx->mapToDevice(_bitmap, &_device, _size);
}

}}}
