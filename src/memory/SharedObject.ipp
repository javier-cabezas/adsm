#ifndef GMAC_MEMORY_SHAREDOBJECT_IPP_
#define GMAC_MEMORY_SHAREDOBJECT_IPP_

#include "Memory.h"

#ifdef USE_VM
#include "Bitmap.h"
#endif

namespace gmac { namespace memory { namespace __impl {

template<typename T>
inline SharedObject<T>::SharedObject(size_t size, void *cpuPtr, T init) :
    StateObject<T>(size, init),
    owner_(&Mode::current()),
    accBlock_(NULL)
{
    gmacError_t ret = gmacSuccess;
    void *accAddr = NULL;
    // Allocate accelerator and host memory
    ret = owner_->malloc(&accAddr, size, (unsigned)paramPageSize);
    if(ret != gmacSuccess) {
        accBlock_ = NULL;
        return;
    }
    StateObject<T>::addr_ = cpuPtr;
    if (cpuPtr != NULL) {
        mapped_ = true;
    } else {
        mapped_ = false;
    }
#ifdef USE_VM
    vm::Bitmap &bitmap = owner_->dirtyBitmap();
    bitmap.newRange(accAddr, size);
#endif
    // Create memory blocks
    accBlock_ = new AcceleratorBlock(*owner_, accAddr, StateObject<T>::size_);
}


template<typename T>
inline SharedObject<T>::~SharedObject()
{
    // TODO Shouldn't this be forbidden?
    if(StateObject<T>::addr_ == NULL) {
        return;
    }
    void *devAddr = accBlock_->addr();
    delete accBlock_;
    owner_->free(devAddr);
#ifdef USE_VM
    vm::Bitmap &bitmap = owner_->dirtyBitmap();
    bitmap.removeRange(devAddr, StateObject<T>::size_);
#endif

    if (mapped_) {
        TRACE(LOCAL,"Unmapping Shared Object %p ("FMT_SIZE" bytes)", StateObject<T>::addr_);
    } else {
        TRACE(LOCAL,"Destroying Shared Object %p ("FMT_SIZE" bytes)", StateObject<T>::addr_);
    }
}

template<typename T>
inline gmacError_t
SharedObject<T>::init()
{
    if(accBlock_ == NULL) {
        return gmacErrorMemoryAllocation;
    }

    if (StateObject<T>::addr_ == NULL) {
#ifdef USE_MMAP
        StateObject<T>::addr_ = StateObject<T>::map(device, StateObject<T>::size_);
#else
        StateObject<T>::addr_ = StateObject<T>::map(NULL, StateObject<T>::size_);
#endif
    }

    if(StateObject<T>::addr_ == NULL) {
        return gmacErrorMemoryAllocation;
    }

    TRACE(LOCAL,"Shared Object %p ("FMT_SIZE" bytes) @ %p initialized", StateObject<T>::addr_, StateObject<T>::size_, accBlock_->addr());
    StateObject<T>::setupSystem();

    return gmacSuccess;
}

template<typename T>
inline void SharedObject<T>::fini()
{
    if (mapped_ == false) {
        StateObject<T>::unmap(StateObject<T>::addr_, StateObject<T>::size_);
    } else {
        // TODO Always R/W?
        Memory::protect(StateObject<T>::addr_, StateObject<T>::size_, GMAC_PROT_READWRITE);
    }
}


template<typename T>
inline void *SharedObject<T>::getAcceleratorAddr(void *addr) const
{
    unsigned offset = (unsigned long)addr - (unsigned long)StateObject<T>::addr_;
    void *ret = accBlock_->addr() + offset;
    return ret;
}

template<typename T>
inline Mode &SharedObject<T>::owner() const
{
    return *owner_;
}

template<typename T>
inline gmacError_t SharedObject<T>::toHost(Block &block) const
{
    unsigned off = (unsigned)(block.addr() - StateObject<T>::addr());
    gmacError_t ret = accBlock_->owner().copyToHost(block.addr(), accBlock_->addr() + off, block.size());
    return ret;
}

template<typename T>
inline gmacError_t SharedObject<T>::toHost(Block &block, unsigned blockOff, size_t count) const
{
    unsigned off = (unsigned)(block.addr() + blockOff - StateObject<T>::addr());
    gmacError_t ret = accBlock_->owner().copyToHost(block.addr() + blockOff, accBlock_->addr() + off, count);

    return ret;
}

template<typename T>
inline gmacError_t SharedObject<T>::toHostPointer(Block &block, unsigned blockOff, void * ptr, size_t count) const
{
    unsigned off = (unsigned)(block.addr() + blockOff - StateObject<T>::addr());
    gmacError_t ret = accBlock_->owner().copyToHost(ptr, accBlock_->addr() + off, count);

    return ret;
}

template<typename T>
inline gmacError_t SharedObject<T>::toHostBuffer(Block &block, unsigned blockOff, IOBuffer &buffer, unsigned bufferOff, size_t count) const
{
    unsigned off = (unsigned)(block.addr() + blockOff - StateObject<T>::addr());
    gmacError_t ret = accBlock_->owner().acceleratorToBuffer(buffer, accBlock_->addr() + off, count, bufferOff);

    return ret;
}

template<typename T>
inline gmacError_t SharedObject<T>::toAccelerator(Block &block) const
{
    unsigned off = unsigned(block.addr() - StateObject<T>::addr());
    gmacError_t ret = accBlock_->owner().copyToAccelerator(accBlock_->addr() + off, block.addr(), block.size());
    return ret;
}

template<typename T>
inline gmacError_t SharedObject<T>::toAccelerator(Block &block, unsigned blockOff, size_t count) const
{
    unsigned off = unsigned(block.addr() + blockOff - StateObject<T>::addr());
    gmacError_t ret = accBlock_->owner().copyToAccelerator(accBlock_->addr() + off, block.addr() + blockOff, count);
    return ret;
}

template<typename T>
inline gmacError_t SharedObject<T>::toAcceleratorFromPointer(Block &block, unsigned blockOff, const void * ptr, size_t count) const
{
    unsigned off = unsigned(block.addr() + blockOff - StateObject<T>::addr());
    gmacError_t ret = accBlock_->owner().copyToAccelerator(accBlock_->addr() + off, ptr, count);

    return ret;
}

template<typename T>
inline gmacError_t SharedObject<T>::toAcceleratorFromBuffer(Block &block, unsigned blockOff, IOBuffer &buffer, unsigned bufferOff, size_t count) const
{
    unsigned off = unsigned(block.addr() + blockOff - StateObject<T>::addr());
    gmacError_t ret = accBlock_->owner().bufferToAccelerator(accBlock_->addr() + off, buffer, count, bufferOff);

    return ret;
}


template<typename T>
gmacError_t SharedObject<T>::free()
{
#ifdef USE_VM
    vm::Bitmap &bitmap = owner_->dirtyBitmap();
    bitmap.removeRange(accBlock_->addr(), StateObject<T>::size_);
#endif
    gmacError_t ret;
    ret = owner_->free(accBlock_->addr());
    delete accBlock_;
    accBlock_ = NULL;
    return ret;
}

template<typename T>
gmacError_t SharedObject<T>::realloc(Mode &mode)
{
    void *device = NULL;
    // Allocate device and host memory
    gmacError_t ret = mode.malloc(&device, StateObject<T>::size_, (unsigned)paramPageSize);
    if(ret != gmacSuccess) {
        StateObject<T>::addr_ = NULL;
        return gmacErrorInsufficientAcceleratorMemory;
    }

    TRACE(LOCAL,"Reallocating object %p -> %p\n", accBlock_->addr(), device);

#ifdef USE_VM
    vm::Bitmap &bitmap = owner_->dirtyBitmap();
    bitmap.newRange(device, StateObject<T>::size_);
#endif
    owner_ = &mode;
    accBlock_ = new AcceleratorBlock(mode, device, StateObject<T>::size_);
    return gmacSuccess;
}

template<typename T>
gmacError_t
SharedObject<T>::memsetAccelerator(gmac::memory::Block &block, unsigned blockOff, int c, size_t count) const
{
    unsigned off = (unsigned)(block.addr() + blockOff - StateObject<T>::addr());
    gmacError_t ret = owner_->memset(accBlock_->addr() + off, c, count);
    return ret;
}

template<typename T>
bool SharedObject<T>::isLocal() const
{
    return true;
}

template<typename T>
bool SharedObject<T>::isInAccelerator() const
{
    return true;
}

}}}

#endif
