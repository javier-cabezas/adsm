#include "Bitmap.h"

#include "kernel/Mode.h"

#include <cmath>
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

#ifdef DEBUG_BITMAP
void Bitmap::dump()
{
    gmac::Context * ctx = Mode::current()->context();
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
#endif

}}}

#endif
