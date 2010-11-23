#include <cmath>
#include <cstring>

#include "Bitmap.h"

#include "core/Mode.h"

#ifdef USE_VM
namespace __impl { namespace memory { namespace vm {

Bitmap::Bitmap(unsigned bits) :
    RWLock("Bitmap"), bitmap_(NULL), dirty_(true), synced_(true), device_(NULL), minAddr_(NULL), maxAddr_(NULL)
{
    shiftPage_ = int(log2(paramPageSize));
    if (paramBitmapChunksPerPage > 1) {
        shiftPage_ -= int(log2(paramBitmapChunksPerPage));
    }
#ifdef BITMAP_BIT
    bitMask_ = (1 << 5) - 1;
    size_ = (1 << (bits - shiftPage_)) / 8;
    bitmap_ = new uint32_t[size_ / sizeof(uint32_t)];
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
    size_ = (1 << (bits - shiftPage_)) * sizeof(T);

    TRACE(LOCAL,"Shift page: %u", shiftPage_);
    TRACE(LOCAL,"Pages: %u", size_ / sizeof(T));

#ifndef USE_HOSTMAP_VM
    bitmap_ = new T[size_ / sizeof(T)];
    memset(bitmap_, 0, size_);
#endif
#endif
}

#ifdef DEBUG_BITMAP
void Bitmap::dump()
{
    core::Context * ctx = Mode::current()->context();
    ctx->invalidate();

    static int idx = 0;
    char path[256];
    sprintf(path, "_bitmap__%d", idx++);
    FILE * file = fopen(path, "w");
    fwrite(bitmap_, 1, size_, file);    
    fclose(file);
}
#endif

}}}

#endif
