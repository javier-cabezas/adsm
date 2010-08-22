#include "Bitmap.h"

#include <kernel/Context.h>

#include <cstring>

#ifdef USE_VM
namespace gmac { namespace memory { namespace vm {

Bitmap::Bitmap(unsigned bits) :
     _bitmap(NULL), _dirty(false), _synced(true), _device(NULL), _minAddr(NULL), _maxAddr(NULL)
{
    _shiftPage = int(log2(paramPageSize));
    if (paramBitmapChunksPerPage > 1) {
        _shiftPage -= int(log2(paramBitmapChunksPerPage));
    }
#ifdef BITMAP_BIT
    _bitMask = (1 << 5) - 1;
    _size = (1 << (bits - _shiftPage)) / 8;
    _bitmap = new uint32_t[_size / sizeof(uint32_t)];
#else
#ifdef BITMAP_BYTE
    typedef uint8_t T;
#else
#ifdef BITMAP_WORD
    typedef uint32_t T;
#else
#error "Bitmap granularity not defined"
#endif
#endif
    _size = (1 << (bits - _shiftPage)) * sizeof(T);

#ifndef USE_HOSTMAP_VM
    _bitmap = new T[_size / sizeof(T)];
    memset(_bitmap, 0, _size);
#endif
#endif
}

Bitmap::~Bitmap()
{
    Context *ctx = Context::current();
#ifdef USE_HOSTMAP_VM
    if (_bitmap != NULL) ctx->hostFree(_bitmap);
#else
    if (_bitmap != NULL) delete [] _bitmap;
    if (_device != NULL) ctx->free(_device);
#endif
}

void Bitmap::allocate()
{
    assertion(_device == NULL);
    Context *ctx = Context::current();
#ifdef USE_HOSTMAP_VM
    ctx->mallocPageLocked((void **)&_bitmap, _size);
    ctx->mapToDevice(_bitmap, &_device, _size);
    memset(_bitmap, 0, _size);
#else
    ctx->malloc((void **)&_device, _size);
#endif
    trace("Allocating dirty bitmap %p -> %p (%zu bytes)", _bitmap, _device, _size);
}

#include "kernel/Context.h"

// TODO remove this shit
//
//
void Bitmap::dump()
{
    gmac::Context * ctx = gmac::Context::current();
    ctx->invalidate();

    static int idx = 0;
    char path[256];
    sprintf(path, "__bitmap_%d", idx++);
    //printf("Writing %p - %zd\n", _bitmap, _size);
    FILE * file = fopen(path, "w");
    fwrite(_bitmap, 1, _size, file);
    //memset(_bitmap, 0, _size);
    fclose(file);
}

void Bitmap::sync()
{
    trace("Syncing Bitmap");
    gmac::Context * ctx = gmac::Context::current();
    ctx->invalidate();
    _synced = true;
}

}}}

#endif
