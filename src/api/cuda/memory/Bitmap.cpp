#include "api/cuda/Accelerator.h"
#include "api/cuda/Mode.h"

#include "memory/Bitmap.h"

#ifdef USE_VM
namespace __impl { namespace memory { namespace vm {

void Bitmap::allocate()
{
    ASSERTION(device_ == NULL);
    cuda::Mode &mode = cuda::Mode::current();
#ifdef USE_HOSTMAP_VM
    mode.hostAlloc((void **)&bitmap_, size_);
    device_ = mode->hostMap(bitmap_);
    memset(bitmap_, 0, size());
    TRACE(LOCAL,"Allocating dirty bitmap (%zu bytes)", size());
#else
    mode.malloc((void **)&device_, size_);
    TRACE(LOCAL,"Allocating dirty bitmap %p -> %p (%zu bytes)", bitmap_, device_, size_);
#endif
}

Bitmap::~Bitmap()
{
    cuda::Mode &mode = __impl::cuda::Mode::current();
    if(device_ != NULL) mode.hostFree(bitmap_);
}

void
Bitmap::syncHost()
{
#ifndef USE_HOSTMAP_VM
    TRACE(LOCAL,"Syncing Bitmap");
    core::Mode &mode = core::Mode::current();

    memory::vm::Bitmap &bitmap = mode.dirtyBitmap();
    TRACE(LOCAL,"Setting dirty bitmap on host: %p -> %p: "FMT_SIZE, (void *) cuda::Accelerator::gpuAddr(bitmap.device()), bitmap.host(), bitmap.size());
    gmacError_t ret;
    //printf("Bitmap toHost\n");
    ret = mode.copyToHost(bitmap.host(), bitmap.device(), bitmap.size());
    CFATAL(ret == gmacSuccess, "Unable to copy back dirty bitmap");
    bitmap.reset();
#endif
}

void
Bitmap::syncDevice()
{
#ifndef USE_HOSTMAP_VM
    TRACE(LOCAL,"Syncing Bitmap");
    cuda::Mode &mode = cuda::Mode::current();

    memory::vm::Bitmap &bitmap = mode.dirtyBitmap();
    TRACE(LOCAL,"Setting dirty bitmap on device: %p -> %p (0x%lx): "FMT_SIZE, (void *) cuda::Accelerator::gpuAddr(bitmap.device()), bitmap.host(), mode.dirtyBitmapDevPtr(), bitmap.size());
    gmacError_t ret;
    ret = mode.copyToAccelerator((void *) mode.dirtyBitmapDevPtr(), &device_, sizeof(void *));
    CFATAL(ret == gmacSuccess, "Unable to set the pointer in the device %p", (void *) mode.dirtyBitmapDevPtr());
    ret = mode.copyToAccelerator((void *) mode.dirtyBitmapShiftPageDevPtr(), &shiftPage_, sizeof(int));
    CFATAL(ret == gmacSuccess, "Unable to set shift page in the device %p", (void *) mode.dirtyBitmapShiftPageDevPtr());
#ifdef BITMAP_BIT
    ret = mode.copyToAccelerator((void *) mode.dirtyBitmapShiftEntryDevPtr(), &shiftEntry_, sizeof(int));
    CFATAL(ret == gmacSuccess, "Unable to set shift entry in the device %p", (void *) mode.dirtyBitmapShiftEntryDevPtr());
#endif

    //printf("Bitmap toHost\n");
    ret = mode.copyToAccelerator(bitmap.device(), bitmap.host(), bitmap.size());
    CFATAL(ret == gmacSuccess, "Unable to copy dirty bitmap to device");

    synced_ = false;
#endif
}

}}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
