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
    _bitmap = new T[_size / sizeof(T)];
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
