#include "Bitmap.h"

#include <kernel/Context.h>

#include <string.h>

namespace gmac { namespace memory { namespace vm {

Bitmap::Bitmap(unsigned bits) :
    __device(NULL)
{
    __pageShift = int(log2(paramPageSize));
    __size = 1 << (bits - __pageShift);
    __bitmap = new uint8_t[__size];
    memset(__bitmap, 0, size());
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
        TRACE("Allocating dirty bitmap (%zu bytes)", size());
        ctx->malloc((void **)&__device, size());
    }
}

}}}
