#include "api/cuda/Accelerator.h"
#include "api/cuda/Mode.h"

#include "memory/Bitmap.h"

#ifdef USE_VM
namespace __impl { namespace memory { namespace vm {

void Bitmap::allocate()
{
    ASSERTION(device_ == NULL);
    cuda::Mode &mode = static_cast<cuda::Mode &>(mode_);
#ifdef USE_HOSTMAP_VM
    mode.hostAlloc((void **)&bitmap_, size_);
    device_ = (uint8_t *) mode.hostMap(bitmap_);
    memset(bitmap_, 0, size());
    TRACE(LOCAL,"Allocating dirty bitmap (%zu bytes)", size());
#else
    mode.malloc((void **)&device_, size_);
    TRACE(LOCAL,"Allocating dirty bitmap %p -> %p (%zu bytes)", bitmap_, device_, size_);
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
    if(device_ != NULL) mode.hostFree(bitmap_);
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
    TRACE(LOCAL,"Setting dirty bitmap on host: %p -> %p: "FMT_SIZE, (void *) cuda::Accelerator::gpuAddr(device()), host(), size());
    gmacError_t ret;
    //printf("Bitmap toHost\n");
    ret = mode.copyToHost(host(), device(), size());
    CFATAL(ret == gmacSuccess, "Unable to copy back dirty bitmap");
    reset();
#endif
}

void
Bitmap::syncDevice()
{
    cuda::Mode &mode = static_cast<cuda::Mode &>(mode_);
    cuda::Accelerator &acc = mode.getAccelerator();

#ifndef USE_MULTI_CONTEXT
    cuda::Mode *last = acc.getLastMode();

    if (last != &mode_) {
        if (last != NULL) {
            Bitmap &lastBitmap = last->dirtyBitmap();
            if (!lastBitmap.synced()) {
                lastBitmap.syncHost();
            }
        }
#endif
        TRACE(LOCAL, "Syncing Bitmap pointers");
        TRACE(LOCAL, "%p -> %p (0x%lx)", host(), (void *) cuda::Accelerator::gpuAddr(device()), mode.dirtyBitmapDevPtr());
        gmacError_t ret;
        ret = mode.copyToAccelerator((void *) mode.dirtyBitmapDevPtr(), &device_, sizeof(void *));
        CFATAL(ret == gmacSuccess, "Unable to set the pointer in the device %p", (void *) mode.dirtyBitmapDevPtr());
        ret = mode.copyToAccelerator((void *) mode.dirtyBitmapShiftPageDevPtr(), &shiftPage_, sizeof(int));
        CFATAL(ret == gmacSuccess, "Unable to set shift page in the device %p", (void *) mode.dirtyBitmapShiftPageDevPtr());

#ifndef USE_MULTI_CONTEXT
    }
#endif

#ifndef USE_HOSTMAP_VM
    if (dirty_) {
        TRACE(LOCAL, "Syncing Bitmap");
        TRACE(LOCAL, "Copying "FMT_SIZE" bytes. ShiftPage: %d", size(), shiftPage_);
        gmacError_t ret;
        ret = mode.copyToAccelerator(device(), host(), size());
        CFATAL(ret == gmacSuccess, "Unable to copy dirty bitmap to device");
    }

    synced_ = false;
#endif

#ifndef USE_MULTI_CONTEXT
    acc.setLastMode(mode);
#endif
}

}}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
