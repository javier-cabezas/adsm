#ifndef GMAC_MEMORY_SHAREDBLOCK_IMPL_H_
#define GMAC_MEMORY_SHAREDBLOCK_IMPL_H_

#include "core/Mode.h"
#include "core/IOBuffer.h"

namespace __impl { namespace memory {

template<typename T>
inline SharedBlock<T>::SharedBlock(Protocol &protocol, core::Mode &owner, hostptr_t hostAddr,
								   hostptr_t shadowAddr, accptr_t acceleratorAddr, size_t size, T init) :
	memory::StateBlock<T>(protocol, hostAddr, shadowAddr, size, init),
	owner_(owner),
	acceleratorAddr_(acceleratorAddr)
{}

template<typename T>
inline SharedBlock<T>::~SharedBlock()
{}

template<typename T>
inline core::Mode &SharedBlock<T>::owner() const
{
	return owner_;
}

template<typename T>
inline accptr_t SharedBlock<T>::acceleratorAddr(const hostptr_t addr) const
{
	ptroff_t offset = ptroff_t(addr - StateBlock<T>::addr_);
    accptr_t ret = acceleratorAddr_ + offset;
	return ret;
}

template<typename T>
inline gmacError_t SharedBlock<T>::toHost() const
{
    gmacError_t ret = gmacSuccess;
#ifdef USE_VM
    vm::BitmapShared &acceleratorBitmap = owner_.acceleratorDirtyBitmap();
    bool inSubGroup = false;
    unsigned groupStart = 0, groupEnd = 0;
    unsigned gaps = 0;
    //fprintf(stderr, "TOHOST: SubBlocks %u\n", Block::getSubBlocks());
    for (unsigned i = 0; i < Block::getSubBlocks(); i++) {
        if (inSubGroup) {
            if (acceleratorBitmap.getAndSetEntry(acceleratorAddr_ + i * SubBlockSize_, vm::BITMAP_UNSET) == vm::BITMAP_SET_ACC) {
                groupEnd = i;
            } else {
                if (vm::costGaps<vm::MODEL_TODEVICE>(SubBlockSize_, gaps + 1, i - groupStart + 1) <
                    vm::cost<vm::MODEL_TODEVICE>(SubBlockSize_, 1)) {
                    gaps++;
                } else {
                    inSubGroup = false;

                    //fprintf(stderr, "TOHOST A: Copying from %u to %u, size %u\n", groupStart, groupEnd, groupEnd - groupStart + 1);
                    ret = owner_.copyToHost(StateBlock<T>::shadow_ + groupStart * SubBlockSize_,
                                            acceleratorAddr_       + groupStart * SubBlockSize_,
                                            (groupEnd - groupStart + 1) * SubBlockSize_);
                    if (ret != gmacSuccess) break;
                }
            }
        } else {
            if (acceleratorBitmap.getAndSetEntry(acceleratorAddr_ + i * SubBlockSize_, vm::BITMAP_UNSET) == vm::BITMAP_SET_ACC) {
                groupStart = groupEnd = i; gaps = 0; inSubGroup = true;
            }
        }
    }
    if (inSubGroup) {
        //fprintf(stderr, "TOHOST B: Copying from %u to %u, size %u\n", groupStart, groupEnd, groupEnd - groupStart + 1);
        ret = owner_.copyToHost(StateBlock<T>::shadow_ + groupStart * SubBlockSize_,
                                acceleratorAddr_       + groupStart * SubBlockSize_,
                                (groupEnd - groupStart + 1) * SubBlockSize_);
    }
#else
    ret = owner_.copyToHost(StateBlock<T>::shadow_, acceleratorAddr_, StateBlock<T>::size_);
#endif
    return ret;
}

template<typename T>
inline gmacError_t SharedBlock<T>::toAccelerator()
{
    gmacError_t ret = gmacSuccess;
#if defined(USE_VM) || defined(USE_SUBBLOCK_TRACKING)
#ifdef USE_SUBBLOCK_TRACKING
    vm::BitmapHost &bitmap = owner_.acceleratorDirtyBitmap();
#else
#ifdef USE_VM
    vm::BitmapShared &bitmap= owner_.acceleratorDirtyBitmap();
#endif
#endif
    bool inSubGroup = false;
    unsigned groupStart = 0, groupEnd = 0;
    unsigned gaps = 0;
    //fprintf(stderr, "TODEVICE: SubBlocks %u\n", Block::getSubBlocks());
    for (unsigned i = 0; i < Block::getSubBlocks(); i++) {
        if (inSubGroup) {
            if (bitmap.getAndSetEntry(acceleratorAddr_ + i * SubBlockSize_, vm::BITMAP_UNSET) == vm::BITMAP_SET_HOST) {
                groupEnd = i;
            } else {
                if (vm::costGaps<vm::MODEL_TODEVICE>(SubBlockSize_, gaps + 1, i - groupStart + 1) <
                    vm::cost<vm::MODEL_TODEVICE>(SubBlockSize_, 1)) {
                    gaps++;
                } else {
                    inSubGroup = false;
                    
                    //fprintf(stderr, "TODEVICE A: Copying from %u to %u, size %u\n", groupStart, groupEnd, groupEnd - groupStart + 1);
                    ret = owner_.copyToAccelerator(acceleratorAddr_       + groupStart * SubBlockSize_,
                                                   StateBlock<T>::shadow_ + groupStart * SubBlockSize_,
                                                   (groupEnd - groupStart + 1) * SubBlockSize_);
                    if (ret != gmacSuccess) break;
                }
            }
        } else {
            if (bitmap.getAndSetEntry(acceleratorAddr_ + i * SubBlockSize_, vm::BITMAP_UNSET) == vm::BITMAP_SET_HOST) {
                groupStart = groupEnd = i; gaps = 0; inSubGroup = true;
            }
        }
    }
    if (inSubGroup) {
        //fprintf(stderr, "TODEVICE B: Copying from %u to %u, size %u\n", groupStart, groupEnd, groupEnd - groupStart + 1);
        ret = owner_.copyToAccelerator(acceleratorAddr_       + groupStart * SubBlockSize_,
                                       StateBlock<T>::shadow_ + groupStart * SubBlockSize_,
                                       (groupEnd - groupStart + 1) * SubBlockSize_);
    }
    Block::resetBitmapStats();
#else
    ret = owner_.copyToAccelerator(acceleratorAddr_, StateBlock<T>::shadow_, StateBlock<T>::size_);
#endif
	return ret;
}

template<typename T>
inline gmacError_t SharedBlock<T>::copyToHost(const hostptr_t src, size_t size, size_t blockOffset) const
{
    ::memcpy(StateBlock<T>::shadow_ + blockOffset, src, size);
    return gmacSuccess;
}

template<typename T>
inline gmacError_t SharedBlock<T>::copyToHost(core::IOBuffer &buffer, size_t size, 
											  size_t bufferOffset, size_t blockOffset) const
{
	::memcpy(StateBlock<T>::shadow_ + blockOffset, buffer.addr() + bufferOffset, size);
	return gmacSuccess;
}

template<typename T>
inline gmacError_t SharedBlock<T>::copyToAccelerator(const hostptr_t src, size_t size,  size_t blockOffset) const
{
    return owner_.copyToAccelerator(acceleratorAddr_ + ptroff_t(blockOffset), src, size);
}

template<typename T>
inline gmacError_t SharedBlock<T>::copyToAccelerator(core::IOBuffer &buffer, size_t size, 
												size_t bufferOffset, size_t blockOffset) const
{
	return owner_.bufferToAccelerator(acceleratorAddr_ + ptroff_t(blockOffset), 
        buffer, size, bufferOffset);
}

template<typename T>
inline gmacError_t SharedBlock<T>::copyFromHost(hostptr_t dst, size_t size, size_t blockOffset) const
{
    ::memcpy(dst, StateBlock<T>::shadow_ + blockOffset, size);
    return gmacSuccess;
}

template<typename T>
inline gmacError_t SharedBlock<T>::copyFromHost(core::IOBuffer &buffer, size_t size, 
												size_t bufferOffset, size_t blockOffset) const
{
	::memcpy(buffer.addr() + bufferOffset, StateBlock<T>::shadow_ + blockOffset, size);
	return gmacSuccess;
}

template<typename T>
inline gmacError_t SharedBlock<T>::copyFromAccelerator(hostptr_t dst, size_t size, size_t blockOffset) const
{
    return owner_.copyToHost(dst, acceleratorAddr_ + ptroff_t(blockOffset), size);
}

template<typename T>
inline gmacError_t SharedBlock<T>::copyFromAccelerator(core::IOBuffer &buffer, size_t size, 
												  size_t bufferOffset, size_t blockOffset) const
{
	return owner_.acceleratorToBuffer(buffer, acceleratorAddr_ + ptroff_t(blockOffset), 
        size, bufferOffset);
}

template<typename T>
inline gmacError_t SharedBlock<T>::hostMemset(int v, size_t size, size_t blockOffset) const
{
    ::memset(StateBlock<T>::shadow_ + blockOffset, v, size);
    return gmacSuccess;
}

template<typename T>
inline gmacError_t SharedBlock<T>::acceleratorMemset(int v, size_t size, size_t blockOffset) const
{
    return owner_.memset(acceleratorAddr_ + ptroff_t(blockOffset), v, size);
}

}}

#endif
