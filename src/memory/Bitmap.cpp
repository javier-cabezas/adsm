#ifdef USE_VM

#include <cstring>

#include "core/Mode.h"

#include "Bitmap.h"

namespace __impl { namespace memory { namespace vm {

#ifdef BITMAP_BIT
const unsigned Bitmap::EntriesPerByte_ = 8;
#else // BITMAP_BYTE
const unsigned Bitmap::EntriesPerByte_ = 1;
#endif

Bitmap::Bitmap(core::Mode &mode, unsigned bits) :
    RWLock("Bitmap"), bits_(bits), mode_(mode), bitmap_(NULL), dirty_(true), minPtr_(NULL), maxPtr_(NULL)
{
    unsigned rootEntries = (1 << bits) >> 32;
    if (rootEntries == 0) rootEntries = 1;
    rootEntries_ = rootEntries;

    bitmap_ = new hostptr_t[rootEntries];
    ::memset(bitmap_, 0, rootEntries * sizeof(hostptr_t));

    shiftBlock_ = int(log2(paramPageSize));
    shiftPage_  = shiftBlock_ - int(log2(paramSubBlocks));

    subBlockSize_ = (paramSubBlocks) - 1;
    subBlockMask_ = (paramSubBlocks) - 1;
    pageMask_     = subBlockSize_ - 1;

    size_    = (1 << (bits - shiftPage_)) / EntriesPerByte_;
#ifdef BITMAP_BIT
    bitMask_ = (1 << 3) - 1;
#endif

    TRACE(LOCAL, "Pages: %u", 1 << (bits - shiftPage_));
    TRACE(LOCAL,"Size : %u", size_);
}

Bitmap::Bitmap(const Bitmap &base) :
    RWLock("Bitmap"),
    bits_(base.bits_),
    mode_(base.mode_),
    bitmap_(base.bitmap_),
    dirty_(true),
    shiftBlock_(base.shiftBlock_),
    shiftPage_(base.shiftPage_),
    subBlockSize_(base.subBlockSize_),
    subBlockMask_(base.subBlockMask_),
    pageMask_(base.pageMask_),
#ifdef BITMAP_BIT
    bitMask_(base.bitMask_),
#endif
    size_(base.size_),
    minEntry_(-1), maxEntry_(-1)
{
}


Bitmap::~Bitmap()
{
    
}

void
Bitmap::cleanUp()
{
    for (unsigned i = minRootEntry_; i <= maxRootEntry_; i++) {
        if (bitmap_[i] != NULL) {
            delete [] bitmap_[i];
        }
    }
    delete [] bitmap_;
}

SharedBitmap::SharedBitmap(core::Mode &mode, unsigned bits) :
    Bitmap(mode, bits), linked_(false), synced_(true), accelerator_(NULL)
{
}

SharedBitmap::SharedBitmap(const Bitmap &host) :
    Bitmap(host), linked_(true), synced_(true), accelerator_(NULL)
{
}

SharedBitmap::~SharedBitmap()
{
}


#ifdef DEBUG_BITMAP
void Bitmap:dump()
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
