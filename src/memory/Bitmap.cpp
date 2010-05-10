#include "Bitmap.h"

#include <kernel/Context.h>

#include <string.h>

namespace gmac { namespace memory { namespace vm {

Bitmap::Bitmap(unsigned bits) :
    __device(NULL)
{
    size_t space = 1 << bits;
    size_t bytes = space / paramPageSize;
    if(space % paramPageSize) bytes++;
    __size = bytes / 32;
    if(bytes % 32) __size++;
    __pageShift = int(log2(paramPageSize));
    __bitmap = new uint32_t[__size];
    memset(__bitmap, 0, __size * sizeof(uint32_t));

}

Bitmap::~Bitmap()
{
    delete[] __bitmap;
    Context *ctx = Context::current();
    if(__device != NULL) ctx->free(__device);
}

void Bitmap::allocate()
{
    Context *ctx = Context::current();
    if(__device == NULL) {
        TRACE("Allocating dirty bitmap");
        ctx->malloc((void **)&__device, __size * sizeof(uint32_t));
    }
}

}}}
