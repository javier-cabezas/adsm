#include "api/cuda/Accelerator.h"
#include "api/cuda/Mode.h"

#include "memory/Bitmap.h"

#ifdef USE_VM
namespace __impl { namespace memory { namespace vm {

void
StoreShared::allocAcc()
{
    accptr_t addr;
    gmacError_t ret = root_.mode_.malloc(addr, size_);
    ASSERTION(ret == gmacSuccess);
}

void
StoreShared::syncToHost(unsigned long startIndex, unsigned long endIndex, size_t elemSize)
{
    return;
#ifndef USE_HOSTMAP_VM
    TRACE(LOCAL,"Syncing SharedBitmap");
    cuda::Mode &mode = static_cast<cuda::Mode &>(root_.mode_);
    gmacError_t ret;
    size_t size = (endIndex - startIndex) * elemSize;
    hostptr_t host = entriesHost_ + startIndex * elemSize;
    accptr_t acc = entriesAcc_ + startIndex * elemSize;
    TRACE(LOCAL,"Setting dirty bitmap on host: %p -> %p: "FMT_SIZE, (void *) acc, host, size);
    ret = mode.copyToHost(host, acc, size);
    CFATAL(ret == gmacSuccess, "Unable to copy to host dirty bitmap node");
#endif
}

void
StoreShared::syncToAccelerator(unsigned long startIndex, unsigned long endIndex, size_t elemSize)
{
    return;
    if (!allocatedAcc_) allocAcc();

    cuda::Mode &mode = static_cast<cuda::Mode &>(root_.mode_);

#ifndef USE_HOSTMAP_VM
    if (isDirty()) {
        TRACE(LOCAL, "Syncing SharedBitmap");
        TRACE(LOCAL, "Copying "FMT_SIZE" bytes", size_);
        gmacError_t ret = gmacSuccess;
        void * entriesAcc = & ((Node **)(void *) entriesAcc_)[startIndex];
        void * entriesHost = & ((Node **)entriesHost_)[startIndex];
        size_t size = endIndex - startIndex * sizeof(Node *);

        ret = mode.copyToAccelerator(accptr_t(entriesAcc), hostptr_t(entriesHost), size);
        CFATAL(ret == gmacSuccess, "Unable to copy dirty bitmap to accelerator");
    }
#endif
}

StoreShared::~StoreShared()
{
    TRACE(LOCAL, "StoreShared constructor");

    if (allocatedAcc_ == true) {
        // TODO: implement accelerator memory deallocation
    }
}


void
BitmapShared::syncToAccelerator()
{
    cuda::Mode &mode = static_cast<cuda::Mode &>(mode_);
    cuda::Accelerator &acc = mode.getAccelerator();

#ifndef USE_MULTI_CONTEXT
    cuda::Mode *last = acc.getLastMode();

    if (last != &mode) {
        // TODO Is this really necessary
        if (last != NULL) {
            BitmapShared &lastBitmap = last->acceleratorDirtyBitmap();
            Node *n = lastBitmap.root_;
            NodeShared *root = (NodeShared *) lastBitmap.root_;
            if (!root->isSynced()) {
                root->syncToHost<Node *>(n->getFirstUsedEntry(), n->getLastUsedEntry());
            }
        }
        TRACE(LOCAL, "Syncing SharedBitmap pointers");
        gmacError_t ret = gmacSuccess;
        accptr_t bitmapAccPtr = mode.dirtyBitmapAccPtr();
        NodeShared *root = (NodeShared *) root_;

        TRACE(LOCAL, "%p -> %p (0x%lx)", root->entriesHost_, (void *) root->entriesAcc_, mode.dirtyBitmapAccPtr());
        void * entriesAcc = root->getAccAddr();
        ret = mode.copyToAccelerator(bitmapAccPtr, hostptr_t(&entriesAcc), sizeof(entriesAcc));
        CFATAL(ret == gmacSuccess, "Unable to set the pointer in the accelerator %p", (void *) mode.dirtyBitmapAccPtr());

        accptr_t bitmapShiftPageAccPtr = mode.dirtyBitmapShiftPageAccPtr();
        ret = mode.copyToAccelerator(bitmapShiftPageAccPtr, hostptr_t(&root->shift_), sizeof(root->shift_));
        CFATAL(ret == gmacSuccess, "Unable to set shift page in the accelerator %p", (void *) mode.dirtyBitmapShiftPageAccPtr());
        if (Bitmap::BitmapLevels_ > 1) {
            accptr_t bitmapShiftL1AccPtr = mode.dirtyBitmapShiftPageAccPtr();
            ret = mode.copyToAccelerator(bitmapShiftPageAccPtr, hostptr_t(&root->shift_), sizeof(root->shift_));
            CFATAL(ret == gmacSuccess, "Unable to set shift page in the accelerator %p", (void *) mode.dirtyBitmapShiftPageAccPtr());
        }


    }

    synced_ = true;
#endif

#ifndef USE_MULTI_CONTEXT
    acc.setLastMode(mode);
#endif
}


#if 0
void SharedBitmap::allocate()
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


void SharedBitmap::allocate()
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

void
SharedBitmap::cleanUp()
{
    cuda::Mode &mode = static_cast<cuda::Mode &>(mode_);
    if(accelerator_ != NULL) mode.hostFree(bitmap_);
#ifndef USE_HOSTMAP_VM
    if(!linked_) {
        Bitmap::cleanUp();
    }
#endif
}

void
SharedBitmap::syncHost()
{
#ifndef USE_HOSTMAP_VM
    TRACE(LOCAL,"Syncing SharedBitmap");
    cuda::Mode &mode = static_cast<cuda::Mode &>(mode_);
    TRACE(LOCAL,"Setting dirty bitmap on host: %p -> %p: "FMT_SIZE, (void *) accelerator(), host(), size());
    gmacError_t ret;
    //printf("SharedBitmap toHost\n");
    ret = mode.copyToHost(host(), accelerator(), size());
    CFATAL(ret == gmacSuccess, "Unable to copy back dirty bitmap");
    reset();
#endif
}

void
SharedBitmap::syncAccelerator()
{
    if (accelerator_ == NULL) allocate();

    cuda::Mode &mode = static_cast<cuda::Mode &>(mode_);
    cuda::Accelerator &acc = dynamic_cast<cuda::Accelerator &>(mode.getAccelerator());

#ifndef USE_MULTI_CONTEXT
    cuda::Mode *last = acc.getLastMode();

    if (last != &mode_) {
        if (last != NULL) {
            SharedBitmap &lastBitmap = last->acceleratorDirtyBitmap();
            if (!lastBitmap.synced()) {
                lastBitmap.syncHost();
            }
        }
        TRACE(LOCAL, "Syncing SharedBitmap pointers");
        TRACE(LOCAL, "%p -> %p (0x%lx)", host(), (void *) accelerator(), mode.dirtyBitmapAccPtr());
        gmacError_t ret = gmacSuccess;
        accptr_t bitmapAccPtr = mode.dirtyBitmapAccPtr();
        accptr_t bitmapShiftPageAccPtr = mode.dirtyBitmapShiftPageAccPtr();
        ret = mode.copyToAccelerator(bitmapAccPtr, hostptr_t(&accelerator_.ptr_), sizeof(accelerator_.ptr_));
        CFATAL(ret == gmacSuccess, "Unable to set the pointer in the accelerator %p", (void *) mode.dirtyBitmapAccPtr());
        ret = mode.copyToAccelerator(bitmapShiftPageAccPtr, hostptr_t(&shiftPage_), sizeof(shiftPage_));
        CFATAL(ret == gmacSuccess, "Unable to set shift page in the accelerator %p", (void *) mode.dirtyBitmapShiftPageAccPtr());
    }

#ifndef USE_HOSTMAP_VM
    if (dirty_) {
        TRACE(LOCAL, "Syncing SharedBitmap");
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
#endif

}}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
