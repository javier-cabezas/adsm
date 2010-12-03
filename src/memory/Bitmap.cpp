#include <cmath>
#include <cstring>

#include "Bitmap.h"

#include "core/Mode.h"

#ifdef USE_VM
namespace __impl { namespace memory { namespace vm {

#ifdef BITMAP_BIT
const unsigned Bitmap::entriesPerByte = 8;
#else // BITMAP_BYTE
const unsigned Bitmap::entriesPerByte = 1;
#endif

Bitmap::Bitmap(core::Mode &mode, unsigned bits) :
    RWLock("Bitmap"), mode_(mode), bitmap_(NULL), dirty_(true), synced_(true), accelerator_(NULL), minEntry_(-1), maxEntry_(-1)
{
    shiftBlock_ = int(log2(paramPageSize));
    shiftPage_  = shiftBlock_ - int(log2(paramBitmapChunksPerPage));

    subBlockSize_ = (paramBitmapChunksPerPage) - 1;
    subBlockMask_ = (paramBitmapChunksPerPage) - 1;
    pageMask_     = subBlockSize_ - 1;

    size_    = (1 << (bits - shiftPage_)) / entriesPerByte;
#ifdef BITMAP_BIT
    bitMask_ = (1 << 3) - 1;
#endif

    TRACE(LOCAL, "Pages: %u", 1 << (bits - shiftPage_));
    TRACE(LOCAL,"Size : %u", size_);

#ifndef USE_HOSTMAP_VM
    bitmap_ = new T[size_];
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
