/* Copyright (c) 2009, 2010, 2011 University of Illinois
                   Universitat Politecnica de Catalunya
                   All rights reserved.

Developed by: IMPACT Research Group / Grup de Sistemes Operatius
              University of Illinois / Universitat Politecnica de Catalunya
              http://impact.crhc.illinois.edu/
              http://gso.ac.upc.edu/

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal with the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
  1. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimers.
  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimers in the
     documentation and/or other materials provided with the distribution.
  3. Neither the names of IMPACT Research Group, Grup de Sistemes Operatius,
     University of Illinois, Universitat Politecnica de Catalunya, nor the
     names of its contributors may be used to endorse or promote products
     derived from this Software without specific prior written permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
WITH THE SOFTWARE.  */

#ifndef GMAC_MEMORY_PROTOCOL_LAZY_BLOCKSTATE_IMPL_H_
#define GMAC_MEMORY_PROTOCOL_LAZY_BLOCKSTATE_IMPL_H_

#include "memory/StateBlock.h"

#include <sstream>

#if defined(USE_SUBBLOCK_TRACKING) || defined(USE_VM)

#include "memory/Memory.h"
#include "memory/vm/Model.h"

namespace __impl {
namespace memory {
namespace protocol {
namespace lazy {

inline
BlockTreeState::BlockTreeState() :
    counter_(0)
{
}

#if 0
inline
void
BlockTreeState::incrementPath(long_t subBlock)
{
    counter_++;
    if      (subBlock  < value_ && left_  != NULL) left_->incrementPath(subBlock);
    else if (subBlock >= value_ && right_ != NULL) right_->incrementPath(subBlock);
}

inline
void
BlockTreeState::reset()
{
    counter_ = 0;
    if      (left_  != NULL) left_->reset();
    else if (right_ != NULL) right_->reset();
}
#endif

inline
BlockTreeInfo::BlockTreeInfo(lazy::Block &block) :
    block_(block)
{ 
    // Initialize subblock state tree (for random access patterns)
    treeState_ = new BlockTreeState[2 * SubBlocks_ - 1];
    treeStateLevels_ = log2(SubBlocks_) + 1;
}

inline
BlockTreeInfo::~BlockTreeInfo()
{
    delete treeState_;
}

inline
void
BlockTreeInfo::reset()
{
    ::memset(treeState_, 0, sizeof(BlockTreeState) * SubBlocks_);
}

inline
BlockTreeInfo::Pair
BlockTreeInfo::increment(unsigned subBlock)
{
    unsigned level     = treeStateLevels_ - 1;
    unsigned levelOff  = 0;
    unsigned levelPos  = subBlock;
    unsigned levelSize = SubBlocks_;
    unsigned children  = 1;
    unsigned inc       = 1;

    unsigned unblock = subBlock;

    Pair ret;
    do {
        // printf("TREE! Level: %u (%u + %u)\n", level, levelOff, levelPos);
        unsigned idx = levelOff + levelPos;
        unsigned &counter = treeState_[idx].counter_;

        // printf("TREE! Counter(pre): %u\n", counter);
        if (counter == children) {
            //FATAL("Inconsistent state");
        } else if (2 * (counter + inc) > children) {
            inc     = children - counter;
            counter = children;

            ret.first  = subBlock & (~(children - 1));
            ret.second = children;
        } else {
            counter += inc;
        }
        // printf("TREE! Counter(post): %u\n", counter);

        levelOff  += levelSize;
        levelSize /= 2;
        levelPos  /= 2;
        children  *= 2;
    } while (level-- > 0);

    return ret;
}

inline
void
BlockTreeInfo::signalWrite(const hostptr_t addr)
{
    long_t currentSubBlock = GetSubBlockIndex(block_.addr(), addr);

    //printf("TREE! <%p> %u\n", block_.addr(), unsigned(currentSubBlock));
    lastUnprotectInfo_ = increment(currentSubBlock);
    //printf("TREE! Result: %u:%u\n", lastUnprotectInfo_.first, lastUnprotectInfo_.second);
}

inline
BlockTreeInfo::Pair
BlockTreeInfo::getUnprotectInfo() 
{
    return lastUnprotectInfo_;
}

inline
StrideInfo::StrideInfo(lazy::Block &block) :
    block_(block)
{
    reset();
}

inline unsigned
StrideInfo::getStridedFaults() const
{
    return stridedFaults_;
}

#define STRIDE_THRESHOLD 4

inline bool
StrideInfo::isStrided() const
{
    /// \todo Do not hardcode the threshold value
    return stridedFaults_ > STRIDE_THRESHOLD;
}

inline hostptr_t
StrideInfo::getFirstAddr() const
{
    return firstAddr_;
}

inline long_t
StrideInfo::getStride() const
{
    return stride_;
}

inline
void
StrideInfo::reset()
{
    stridedFaults_ = 0;
    stride_ = 0;
}

inline void
StrideInfo::signalWrite(hostptr_t addr)
{
    if (stridedFaults_ == 0) {
        stridedFaults_ = 1;
        firstAddr_ = addr;
        //printf("STRIDE 1\n");
    } else if (stridedFaults_ == 1) {
        stride_ = addr - lastAddr_;
        stridedFaults_ = 2;
        //printf("STRIDE 2\n");
    } else {
        if (addr == lastAddr_ + stride_) {
            stridedFaults_++;
            //printf("STRIDE 3a\n");
        } else {
            stride_ = addr - lastAddr_;
            stridedFaults_ = 2;
            //printf("STRIDE 3b\n");
        }
    }
    lastAddr_ = addr;
}

inline
void
BlockState::setSubBlock(const hostptr_t addr, ProtocolState state)
{
    setSubBlock(GetSubBlockIndex(block_.addr(), addr), state);
}

inline
void
BlockState::setSubBlock(long_t subBlock, ProtocolState state)
{
#ifdef USE_VM
    vm::Bitmap &bitmap = block_.owner().getBitmap();
    bitmap.setEntry(block_.acceleratorAddr(block_.addr() + subBlock * SubBlockSize_), state);
#else
    subBlockState_[subBlock] = uint8_t(state);
#endif
}

inline
void
BlockState::setAll(ProtocolState state)
{
#ifdef USE_VM
    vm::Bitmap &bitmap = block_.owner().getBitmap();
    bitmap.setEntryRange(block_.acceleratorAddr(block_.addr()), block_.size(), state);
#else
    ::memset(&subBlockState_[0], uint8_t(state), subBlockState_.size() * sizeof(uint8_t));
#endif
}

inline
BlockState::BlockState(lazy::State init, lazy::Block &block) :
    common::BlockState<lazy::State>(init),
    block_(block),
#ifdef USE_VM
    subBlocks_(block_.size()/SubBlockSize_ + ((block_.size() % SubBlockSize_ == 0)? 0: 1)),
#else
    subBlockState_(block_.size()/SubBlockSize_ + ((block_.size() % SubBlockSize_ == 0)? 0: 1)),
#endif
#ifdef DEBUG
    subBlockFaults_(block_.size()/SubBlockSize_ + ((block_.size() % SubBlockSize_ == 0)? 0: 1)),
    transfersToAccelerator_(block_.size()/SubBlockSize_ + ((block_.size() % SubBlockSize_ == 0)? 0: 1)),
    transfersToHost_(block_.size()/SubBlockSize_ + ((block_.size() % SubBlockSize_ == 0)? 0: 1)),
#endif
    strideInfo_(block),
    treeInfo_(block),
    faults_(0)
{ 
    // Initialize subblock states
#ifndef USE_VM
    setAll(init);
#endif

#ifdef DEBUG
    ::memset(&subBlockFaults_[0],         0, subBlockFaults_.size() * sizeof(long_t));
    ::memset(&transfersToAccelerator_[0], 0, transfersToAccelerator_.size() * sizeof(long_t));
    ::memset(&transfersToHost_[0],        0, transfersToHost_.size() * sizeof(long_t));
#endif
}

inline hostptr_t
BlockState::getSubBlockAddr(const hostptr_t addr) const
{
    return GetSubBlockAddr(block_.addr(), addr);
}

inline hostptr_t
BlockState::getSubBlockAddr(unsigned index) const
{
    return GetSubBlockAddr(block_.addr(), block_.addr() + index * SubBlockSize_);
}

inline unsigned
BlockState::getSubBlocks() const
{
    unsigned subBlocks = block_.size()/SubBlockSize_;
    if (block_.size() % SubBlockSize_ != 0) subBlocks++;
    return subBlocks;
}

inline size_t
BlockState::getSubBlockSize() const
{
    return block_.size() < SubBlockSize_? block_.size(): SubBlockSize_;
}

inline gmacError_t
BlockState::syncToAccelerator()
{
    gmacError_t ret = gmacSuccess;

    unsigned groupStart = 0, groupEnd = 0;
    unsigned gaps = 0;
    bool inGroup = false;

#ifdef USE_VM
    vm::Bitmap &bitmap = block_.owner().getBitmap();
    for (unsigned i = 0; i != subBlocks_; i++) {
        if (bitmap.getEntry<ProtocolState>(block_.acceleratorAddr() + i * SubBlockSize_) == lazy::Dirty) {
#else
    for (unsigned i = 0; i != subBlockState_.size(); i++) {
        if (subBlockState_[i] == lazy::Dirty) {
#endif
#ifdef DEBUG
            transfersToAccelerator_[i]++;
#endif
            if (!inGroup) {
                groupStart = i;
                inGroup = true;
            }
            setSubBlock(i, lazy::ReadOnly);
            groupEnd = i;
        } else if (inGroup) {
            if (vm::costGaps<vm::MODEL_TODEVICE>(SubBlockSize_, gaps + 1, i - groupStart + 1) <
                    vm::cost<vm::MODEL_TODEVICE>(SubBlockSize_, 1)) {
                gaps++;
            } else {
                ret = block_.toAccelerator(groupStart * SubBlockSize_, SubBlockSize_ * (groupEnd - groupStart + 1) );
                gaps = 0;
                inGroup = false;
                if (ret != gmacSuccess) break;
            }
        }
    }

    if (inGroup) {
        ret = block_.toAccelerator(groupStart * SubBlockSize_,
                                   SubBlockSize_ * (groupEnd - groupStart + 1));
    }

    reset();
	return ret;
}

inline gmacError_t
BlockState::syncToHost()
{
#ifndef USE_VM
    gmacError_t ret = block_.toHost();
#else
    gmacError_t ret = gmacSuccess;

    unsigned groupStart = 0, groupEnd = 0;
    unsigned gaps = 0;
    bool inGroup = false;

    vm::Bitmap &bitmap = block_.owner().getBitmap();
    for (unsigned i = 0; i != subBlocks_; i++) {
        if (bitmap.getEntry<ProtocolState>(block_.acceleratorAddr() + i * SubBlockSize_) == lazy::Invalid) {
#ifdef DEBUG
            transfersToHost_[i]++;
#endif
            if (!inGroup) {
                groupStart = i;
                inGroup = true;
            }
            setSubBlock(i, lazy::ReadOnly);
            groupEnd = i;
        } else if (inGroup) {
            if (vm::costGaps<vm::MODEL_TOHOST>(SubBlockSize_, gaps + 1, i - groupStart + 1) <
                    vm::cost<vm::MODEL_TOHOST>(SubBlockSize_, 1)) {
                gaps++;
            } else {
                ret = block_.toHost(groupStart * SubBlockSize_, SubBlockSize_ * (groupEnd - groupStart + 1) );
                gaps = 0;
                inGroup = false;
                if (ret != gmacSuccess) break;
            }
        }
    }

    if (inGroup) {
        ret = block_.toHost(groupStart * SubBlockSize_,
                            SubBlockSize_ * (groupEnd - groupStart + 1));
    }

#endif

    return ret;
}

inline void
BlockState::read(const hostptr_t addr)
{
    return;
}

inline void
BlockState::write(const hostptr_t addr)
{
    long_t currentSubBlock = GetSubBlockIndex(block_.addr(), addr);

    faults_++;

    setSubBlock(currentSubBlock, lazy::Dirty);

#ifdef DEBUG
    subBlockFaults_[currentSubBlock]++;
#endif
    strideInfo_.signalWrite(addr);

    if (strideInfo_.isStrided()) {
        for (hostptr_t cur = strideInfo_.getFirstAddr(); cur >= block_.addr() &&
                                                         cur < (block_.addr() + block_.size());
            cur += strideInfo_.getStride()) {
            long_t subBlock = GetSubBlockIndex(block_.addr(), cur);
            setSubBlock(subBlock, lazy::Dirty);
        }
    } else {
        treeInfo_.signalWrite(addr);
        if (faults_ > STRIDE_THRESHOLD) {
            BlockTreeInfo::Pair info = treeInfo_.getUnprotectInfo();

            for (unsigned i = info.first; i < info.first + info.second; i++) {
                setSubBlock(i, lazy::Dirty);
            }
        }
    }
}

inline bool
BlockState::is(ProtocolState state) const
{
#ifdef USE_VM
    vm::Bitmap &bitmap = block_.owner().getBitmap();
    return bitmap.isAnyInRange(block_.acceleratorAddr(block_.addr()), block_.size(), state);
#else
    for (unsigned i = 0; i < subBlockState_.size(); i++) {
        if (subBlockState_[i] == state) return true;
    }

    return false;
#endif
}

inline void
BlockState::reset()
{
    if (faults_ > 0) {
        setAll(lazy::ReadOnly);
        faults_ = 0;

        strideInfo_.reset();
        treeInfo_.reset();
    }
}

inline int
BlockState::unprotect()
{
    int ret = 0;
#if 0
    ret = Memory::protect(block_.addr(), block_.size(), prot);
#else
    unsigned start = 0;
    unsigned size = 0;
#ifdef USE_VM
    vm::Bitmap &bitmap = block_.owner().getBitmap();
    for (unsigned i = 0; i < subBlocks_; i++) {
        ProtocolState state = bitmap.getEntry<ProtocolState>(block_.acceleratorAddr() + i * SubBlockSize_);
#else
    for (unsigned i = 0; i < subBlockState_.size(); i++) {
        ProtocolState state = ProtocolState(subBlockState_[i]);
#endif
        if (state == lazy::Dirty) {
            if (size == 0) start = i;
            size++;
        } else if (size > 0) {
            ret = Memory::protect(getSubBlockAddr(start), SubBlockSize_ * size, GMAC_PROT_READWRITE);
            if (ret < 0) break;
            size = 0;
        }
    }
    if (size > 0) {
        ret = Memory::protect(getSubBlockAddr(start), SubBlockSize_ * size, GMAC_PROT_READWRITE);
    }
#endif
    return ret;
}

inline int
BlockState::protect(GmacProtection prot)
{
    int ret = 0;
#if 1
    ret = Memory::protect(block_.addr(), block_.size(), prot);
#else
    for (unsigned i = 0; i < subBlockState_.size(); i++) {
        if (subBlockState_[i] == lazy::Dirty) {
            ret = Memory::protect(getSubBlockAddr(i), SubBlockSize_, prot);
            if (ret < 0) break;
        }
    }
#endif
    return ret;
}

inline
void
BlockState::acquired()
{
#ifdef DEBUG
    //::memset(&subBlockFaults_[0], 0, subBlockFaults_.size() * sizeof(long_t));
#endif
}

inline
void
BlockState::released()
{
#ifdef DEBUG
    //::memset(&subBlockFaults_[0], 0, subBlockFaults_.size() * sizeof(long_t));
#endif
}

inline
gmacError_t
BlockState::dump(std::ostream &stream, common::Statistic stat)
{
#ifdef DEBUG
    if (stat == common::PAGE_FAULTS) {
        for (unsigned i = 0; i < subBlockFaults_.size(); i++) {
            std::ostringstream oss;
            oss << subBlockFaults_[i] << " ";

            stream << oss.str();
        }
        ::memset(&subBlockFaults_[0], 0, subBlockFaults_.size() * sizeof(long_t));
    } else if (stat == common::PAGE_TRANSFERS_TO_ACCELERATOR) {
        for (unsigned i = 0; i < transfersToAccelerator_.size(); i++) {
            std::ostringstream oss;
            oss << transfersToAccelerator_[i] << " ";

            stream << oss.str();
        }
        ::memset(&transfersToAccelerator_[0], 0, transfersToAccelerator_.size() * sizeof(long_t));
    } else if (stat == common::PAGE_TRANSFERS_TO_HOST) {
        for (unsigned i = 0; i < transfersToHost_.size(); i++) {
            std::ostringstream oss;
            oss << transfersToHost_[i] << " ";

            stream << oss.str();
        }
        ::memset(&transfersToHost_[0], 0, transfersToHost_.size() * sizeof(long_t));
    }
#endif
    return gmacSuccess;
}

}}}}

#else

namespace __impl {
namespace memory {
namespace protocol {
namespace lazy {

inline
BlockState::BlockState(ProtocolState init, lazy::Block &block) :
    common::BlockState<lazy::State>(init),
    block_(block),
    faults_(0),
    transfersToAccelerator_(0),
    transfersToHost_(0)
{
}

inline
gmacError_t
BlockState::syncToAccelerator()
{
    transfersToAccelerator_++;
    return block_.toAccelerator();
}

inline
gmacError_t
BlockState::syncToHost()
{
    transfersToHost_++;
    return block_.toHost();
}

inline
void
BlockState::read(const hostptr_t /*addr*/)
{
}

inline
void
BlockState::write(const hostptr_t /*addr*/)
{
    faults_++;
}

inline
bool
BlockState::is(ProtocolState state) const
{
    return state_ == state;
}

inline
int
BlockState::protect(GmacProtection prot)
{
    return Memory::protect(block_.addr(), block_.size(), prot);
}

inline
int
BlockState::unprotect()
{
    return Memory::protect(block_.addr(), block_.size(), GMAC_PROT_READWRITE);
}

inline
void
BlockState::acquired()
{
    faults_ = 0;
}

inline
void
BlockState::released()
{
}

inline
gmacError_t
BlockState::dump(std::ostream &stream, common::Statistic stat)
{
#ifdef DEBUG
    if (stat == common::PAGE_FAULTS) {
        std::ostringstream oss;
        oss << faults_ << " ";
        stream << oss.str();

        faults_ = 0;
    } else if (stat == common::PAGE_TRANSFERS_TO_ACCELERATOR) {
        std::ostringstream oss;
        oss << transfersToAccelerator_ << " ";
        stream << oss.str();

        transfersToAccelerator_ = 0;
    } else if (stat == common::PAGE_TRANSFERS_TO_HOST) {
        std::ostringstream oss;
        oss << transfersToHost_ << " ";
        stream << oss.str();

        transfersToHost_ = 0;
    }
#endif
    return gmacSuccess;
}

}}}}

#endif

#endif /* BLOCKSTATE_IMPL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
