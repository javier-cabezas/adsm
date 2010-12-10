#include "api/cuda/Accelerator.h"
#include "api/cuda/Mode.h"

#include "memory/Bitmap.h"

#ifdef USE_VM
namespace __impl { namespace memory { namespace vm {

void Bitmap::allocate()
{
    ASSERTION(accelerator_ == NULL);
    cuda::Mode &mode = static_cast<cuda::Mode &>(mode_);
#ifdef USE_HOSTMAP_VM
    mode.hostAlloc((void **)&bitmap_, size_);
    accelerator_ = (uint8_t *) mode.hostMap(bitmap_);
    memset(bitmap_, 0, size());
    TRACE(LOCAL,"Allocating dirty bitmap (%zu bytes)", size());
#else
    mode.malloc(&accelerator_, size_);
    TRACE(LOCAL,"Allocating dirty bitmap %p -> %p (%zu bytes)", bitmap_, (void *) accelerator_, size_);
#endif
}

Bitmap::~Bitmap()
{
}

void
Bitmap::cleanUp()
{
#ifdef USE_HOSTMAP_VM
    cuda::Mode &mode = static_cast<cuda::Mode &>(mode_);
    if(accelerator_ != NULL) mode.hostFree(bitmap_);
#else
    delete [] bitmap_;
#endif
}

void
Bitmap::syncHost()
{
#ifndef USE_HOSTMAP_VM
    TRACE(LOCAL,"Syncing Bitmap");
    cuda::Mode &mode = static_cast<cuda::Mode &>(mode_);
    TRACE(LOCAL,"Setting dirty bitmap on host: %p -> %p: "FMT_SIZE, (void *) accelerator(), host(), size());
    gmacError_t ret;
    //printf("Bitmap toHost\n");
    ret = mode.copyToHost(host(), accelerator(), size());
    CFATAL(ret == gmacSuccess, "Unable to copy back dirty bitmap");
    reset();
#endif
}

void
Bitmap::syncAccelerator()
{
    if (accelerator_ == NULL) allocate();

    cuda::Mode &mode = static_cast<cuda::Mode &>(mode_);
    cuda::Accelerator &acc = dynamic_cast<cuda::Accelerator &>(mode.getAccelerator());

#ifndef USE_MULTI_CONTEXT
    cuda::Mode *last = acc.getLastMode();

    if (last != &mode_) {
        if (last != NULL) {
            Bitmap &lastBitmap = last->dirtyBitmap();
            if (!lastBitmap.synced()) {
                lastBitmap.syncHost();
            }
        }
        TRACE(LOCAL, "Syncing Bitmap pointers");
        TRACE(LOCAL, "%p -> %p (0x%lx)", host(), (void *) accelerator(), mode.dirtyBitmapAccPtr());
        gmacError_t ret = gmacSuccess;
        ret = mode.copyToAccelerator((void *) mode.dirtyBitmapAccPtr(), hostptr_t(&accelerator_.ptr_), sizeof(accelerator_.ptr_));
        CFATAL(ret == gmacSuccess, "Unable to set the pointer in the accelerator %p", (void *) mode.dirtyBitmapAccPtr());
        ret = mode.copyToAccelerator((void *) mode.dirtyBitmapShiftPageAccPtr(), hostptr_t(&shiftPage_), sizeof(shiftPage_));
        CFATAL(ret == gmacSuccess, "Unable to set shift page in the accelerator %p", (void *) mode.dirtyBitmapShiftPageAccPtr());
    }

#ifndef USE_HOSTMAP_VM
    if (dirty_) {
        TRACE(LOCAL, "Syncing Bitmap");
        TRACE(LOCAL, "Copying "FMT_SIZE" bytes. ShiftPage: %d", size(), shiftPage_);
        gmacError_t ret = gmacSuccess;
        ret = mode.copyToAccelerator(accelerator(), host(), size());
        CFATAL(ret == gmacSuccess, "Unable to copy dirty bitmap to accelerator");
    }

    synced_ = false;
#endif

#ifndef USE_MULTI_CONTEXT
    acc.setLastMode(mode);
#endif
#endif
}

}}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
