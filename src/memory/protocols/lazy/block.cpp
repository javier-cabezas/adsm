/* Copyright (c) 2009-2011 University of Illinois
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

#include <sstream>

#if defined(USE_SUBBLOCK_TRACKING) || defined(USE_VM)

#ifdef USE_VM
#include "core/hpe/Mode.h"
#include "core/hpe/process.h"
#endif

#include "memory/address_space.h"

#include "block.h"

namespace __impl {
namespace memory {
namespace protocols {
namespace lazy_types {

BlockTreeState::BlockTreeState() :
    counter_(0)
{
}

BlockTreeInfo::BlockTreeInfo(lazy_types::block &block) :
    block_(block)
{ 
    // Initialize subblock state tree (for random access patterns)
    treeState_ = new BlockTreeState[2 * block.getSubBlocks() - 1];
    bool isPower;
    unsigned levels = log2(block.getSubBlocks(), isPower) + 1;
    if (isPower == false) {
        levels++;
    }
    treeStateLevels_ = levels;
}

BlockTreeInfo::~BlockTreeInfo()
{
    delete treeState_;
}

void
BlockTreeInfo::reset()
{
    ::memset(treeState_, 0, sizeof(BlockTreeState) * block_.getSubBlocks());
}

BlockTreeInfo::Pair
BlockTreeInfo::increment(unsigned subBlock)
{
    unsigned level     = treeStateLevels_ - 1;
    unsigned levelOff  = 0;
    unsigned levelPos  = subBlock;
    unsigned levelSize = block_.getSubBlocks();
    unsigned children  = 1;
    unsigned inc       = 1;

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
            if (ret.first + ret.second > block_.getSubBlocks()) {
                ret.second = block_.getSubBlocks() - ret.first;
            }
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

void
BlockTreeInfo::signal_write(host_const_ptr addr)
{
    long_t currentSubBlock = GetSubBlockIndex(block_.addr(), addr);

    //printf("TREE! <%p> %u\n", block_.addr(), unsigned(currentSubBlock));
    lastUnprotectInfo_ = increment(currentSubBlock);
    //printf("TREE! Result: %u:%u\n", lastUnprotectInfo_.first, lastUnprotectInfo_.second);
}

BlockTreeInfo::Pair
BlockTreeInfo::getUnprotectInfo() 
{
    return lastUnprotectInfo_;
}

StrideInfo::StrideInfo(lazy_types::block &block) :
    block_(block)
{
    reset();
}

unsigned
StrideInfo::getStridedFaults() const
{
    return stridedFaults_;
}

#define STRIDE_THRESHOLD 4

bool
StrideInfo::isStrided() const
{
    /// \todo Do not hardcode the threshold value
    return stridedFaults_ > STRIDE_THRESHOLD;
}

host_ptr
StrideInfo::getFirstAddr() const
{
    return firstAddr_;
}

long_t
StrideInfo::getStride() const
{
    return stride_;
}

void
StrideInfo::reset()
{
    stridedFaults_ = 0;
    stride_ = 0;
}

void
StrideInfo::signal_write(host_ptr addr)
{
    if (stridedFaults_ == 0) {
        stridedFaults_ = 1;
        firstAddr_ = addr;
        //printf("STRIDE 1\n");
    } else if (stridedFaults_ == 1) {
        stride_ = addr - lastAddr_;
        stridedFaults_ = 2;
        //printf("STRIDE 2: %lu\n", stride_);
    } else {
        if (addr == lastAddr_ + stride_) {
            stridedFaults_++;
            //printf("STRIDE 3a\n");
        } else {
            stride_ = addr - lastAddr_;
            stridedFaults_ = 2;
            //printf("STRIDE 3b: %lu\n", stride_);
        }
    }
    lastAddr_ = addr;
}

block &
block_state::block()
{
	//return reinterpret_cast<Block &>(*this);
	return *(block *)this;
}

const block &
block_state::block() const
{
	//return reinterpret_cast<const Block &>(*this);
	return *(const block *)this;
}

void
block_state::setSubBlock(host_const_ptr addr, protocol_state state)
{
    setSubBlock(GetSubBlockIndex(block().addr(), addr), state);
}

#ifdef USE_VM
#define GetMode() (*(core::hpe::Mode *)(void *) &block_.owner(getProcess().getCurrentMode()))
#define GetBitmap() GetMode().getBitmap()
#endif

void
block_state::setSubBlock(long_t subBlock, protocol_state state)
{
#ifdef USE_VM
    vm::Bitmap &bitmap = GetBitmap();
    bitmap.setEntry(block().acceleratorAddr(GetMode(), block().addr() + subBlock * SubBlockSize_), state);
#else
    subBlockState_[subBlock] = uint8_t(state);
#endif
}

void
block_state::setAll(protocol_state state)
{
#ifdef USE_VM
    vm::Bitmap &bitmap = GetBitmap();
    bitmap.setEntryRange(block().acceleratorAddr(GetMode(), block().addr()), block().size(), state);
#else
    ::memset(&subBlockState_[0], uint8_t(state), subBlockState_.size() * sizeof(uint8_t));
#endif
}

block_state::block_state(lazy_types::State init) :
    common::block_state<lazy_types::State>(init),
    subBlocks_(block().size()/SubBlockSize_ + ((block().size() % SubBlockSize_ == 0)? 0: 1)),
    subBlockState_(subBlocks_),
#ifdef DEBUG
    subBlockFaultsRead_(subBlocks_),
    subBlockFaultsWrite_(subBlocks_),
    transfersToAccelerator_(subBlocks_),
    transfersToHost_(subBlocks_),
#endif
    strideInfo_(block()),
    treeInfo_(block()),
    faultsRead_(0),
    faultsWrite_(0)
{ 
    // Initialize subblock states
#ifndef USE_VM
    setAll(init);
#endif

#ifdef DEBUG
    ::memset(&subBlockFaultsRead_[0],     0, subBlockFaultsRead_.size() * sizeof(long_t));
    ::memset(&subBlockFaultsWrite_[0],    0, subBlockFaultsWrite_.size() * sizeof(long_t));
    ::memset(&transfersToAccelerator_[0], 0, transfersToAccelerator_.size() * sizeof(long_t));
    ::memset(&transfersToHost_[0],        0, transfersToHost_.size() * sizeof(long_t));
#endif
}

#if 0
common::block_state<lazy_types::State>::protocol_state
block_state::get_state(host_ptr addr) const
{
    return protocol_state(subBlockState_[GetSubBlockIndex(block().addr(), addr)]);
}
#endif

void
block_state::set_state(protocol_state state, host_ptr addr)
{
    if (addr == NULL) {
        setAll(state);
        state_ = state;
    } else {
        if (state == lazy_types::Dirty) {
            state_ = lazy_types::Dirty;
        } else if (state == lazy_types::ReadOnly) {
            if (state_ != lazy_types::Dirty) state_ = lazy_types::ReadOnly;
        } else {
            FATAL("Wrong state transition");
        }

        subBlockState_[GetSubBlockIndex(block().addr(), addr)] = state;
    }
}

host_ptr
block_state::getSubBlockAddr(host_const_ptr addr) const
{
    return GetSubBlockAddr(block().addr(), addr);
}

host_ptr
block_state::getSubBlockAddr(unsigned index) const
{
    return GetSubBlockAddr(block().addr(), block().addr() + index * SubBlockSize_);
}

unsigned
block_state::getSubBlocks() const
{
    return subBlocks_;
}

size_t
block_state::getSubBlockSize() const
{
    return block().size() < SubBlockSize_? block().size(): SubBlockSize_;
}

hal::event_ptr
block_state::sync_to_device(gmacError_t &err)
{
    hal::event_ptr ret;
    err = gmacSuccess;

    unsigned groupStart = 0, groupEnd = 0;
    unsigned gaps = 0;
    bool inGroup = false;

    TRACE(LOCAL, "Transfer block to accelerator: %p", block().addr());

#ifdef USE_VM
    vm::Bitmap &bitmap = GetBitmap();
#endif
    for (unsigned i = 0; i != subBlocks_; i++) {
#ifdef USE_VM
        if (bitmap.getEntry<protocol_state>(block().acceleratorAddr(GetMode()) + i * SubBlockSize_) == lazy_types::Dirty) {
#else
        if (subBlockState_[i] == lazy_types::Dirty) {
#endif
            if (!inGroup) {
                groupStart = i;
                inGroup = true;
            }
            setSubBlock(i, lazy_types::ReadOnly);
            groupEnd = i;
        } else if (inGroup) {
            if (vm::costGaps<vm::MODEL_TODEVICE>(SubBlockSize_, gaps + 1, i - groupStart + 1) <
                    vm::cost<vm::MODEL_TODEVICE>(SubBlockSize_, 1)) {
                gaps++;
            } else {
                size_t sizeTransfer = SubBlockSize_ * (groupEnd - groupStart + 1);
                if (sizeTransfer > block().size()) sizeTransfer = block().size();
                ret = block().get_owner()->copy_async(block().get_device_addr()        + groupStart * SubBlockSize_,
                                                      hal::ptr(block().get_shadow()) + groupStart * SubBlockSize_,
                                                      sizeTransfer, err);
#ifdef DEBUG
                for (unsigned j = groupStart; j <= groupEnd; j++) { 
                    transfersToAccelerator_[j]++;
                }
#endif
                gaps = 0;
                inGroup = false;
                if (err != gmacSuccess) break;
            }
        }
    }

    if (err == gmacSuccess && inGroup) {
        size_t sizeTransfer = SubBlockSize_ * (groupEnd - groupStart + 1);
        if (sizeTransfer > block().size()) sizeTransfer = block().size();
        ret = block().get_owner()->copy_async(block().get_device_addr()        + groupStart * SubBlockSize_,
                                              hal::ptr(block().get_shadow()) + groupStart * SubBlockSize_,
                                              sizeTransfer, err);
                                    
#ifdef DEBUG
        for (unsigned j = groupStart; j <= groupEnd; j++) { 
            transfersToAccelerator_[j]++;
        }
#endif

    }

	return ret;
}

hal::event_ptr
block_state::sync_to_host(gmacError_t &err)
{
    TRACE(LOCAL, "Transfer block to host: %p", block().addr());

#ifndef USE_VM
    hal::event_ptr ret;
    ret = block().get_owner()->copy_async(hal::ptr(block().get_shadow()),
                                          block().get_device_addr(),
                                          block().size(), err);
#ifdef DEBUG
    for (unsigned i = 0; i < subBlockState_.size(); i++) {
        transfersToHost_[i]++;
    }
#endif

#else
    gmacError_t ret = gmacSuccess;

    unsigned groupStart = 0, groupEnd = 0;
    unsigned gaps = 0;
    bool inGroup = false;

    vm::Bitmap &bitmap = GetBitmap();
    for (unsigned i = 0; i != subBlocks_; i++) {
        if (bitmap.getEntry<protocol_state>(block().acceleratorAddr(GetMode()) + i * SubBlockSize_) == lazy_types::Invalid) {
#ifdef DEBUG
            transfersToHost_[i]++;
#endif
            if (!inGroup) {
                groupStart = i;
                inGroup = true;
            }
            setSubBlock(i, lazy_types::ReadOnly);
            groupEnd = i;
        } else if (inGroup) {
            if (vm::costGaps<vm::MODEL_TOHOST>(SubBlockSize_, gaps + 1, i - groupStart + 1) <
                    vm::cost<vm::MODEL_TOHOST>(SubBlockSize_, 1)) {
                gaps++;
            } else {
                ret = block().get_owner()->copy_async(hal::ptr(block().get_shadow()) + groupStart * SubBlockSize_,
                                                      block().get_device_addr()        + groupStart * SubBlockSize_,
                                                      SubBlockSize_ * (groupEnd - groupStart + 1), err);
                gaps = 0;
                inGroup = false;
                if (err != gmacSuccess) break;
            }
        }
    }

    if (err == gmacSuccess && inGroup) {
        ret = block().get_owner()->copy_async(hal::ptr(block().get_shadow()) + groupStart * SubBlockSize_,
                                              block().get_device_addr()        + groupStart * SubBlockSize_,
                                              SubBlockSize_ * (groupEnd - groupStart + 1), err);
    }

#endif

    return ret;
}

void
block_state::read(host_const_ptr addr)
{
    long_t currentSubBlock = GetSubBlockIndex(block().addr(), addr);
    faultsRead_++;
    faultsCacheRead_++;

    setSubBlock(currentSubBlock, lazy_types::ReadOnly);
#ifdef DEBUG
    subBlockFaultsRead_[currentSubBlock]++;
#endif
    TRACE(LOCAL, "");

    return;
}

void
block_state::writeStride(host_const_ptr addr)
{
    strideInfo_.signal_write(addr);
    if (strideInfo_.isStrided()) {
        for (host_ptr cur = strideInfo_.getFirstAddr(); cur >= block().addr() &&
                cur < (block().addr() + block().size());
                cur += strideInfo_.getStride()) {
            long_t subBlock = GetSubBlockIndex(block().addr(), cur);
            setSubBlock(subBlock, lazy_types::Dirty);
        }
    }
}

void
block_state::writeTree(host_const_ptr addr)
{
    treeInfo_.signal_write(addr);
    BlockTreeInfo::Pair info = treeInfo_.getUnprotectInfo();

    for (unsigned i = info.first; i < info.first + info.second; i++) {
        setSubBlock(i, lazy_types::Dirty);
    }
}

void
block_state::write(host_const_ptr addr)
{
    long_t currentSubBlock = GetSubBlockIndex(block().addr(), addr);

    faultsWrite_++;
    faultsCacheWrite_++;

    setSubBlock(currentSubBlock, lazy_types::Dirty);

#ifdef DEBUG
    subBlockFaultsWrite_[currentSubBlock]++;
#endif

    if (subBlockState_.size() > STRIDE_THRESHOLD) {
        if (util::params::ParamSubBlockStride) {
            writeStride(addr);
            if (util::params::ParamSubBlockTree && !strideInfo_.isStrided()) {
                writeTree(addr);
            }
        } else if (util::params::ParamSubBlockTree) {
            writeTree(addr);
        }
    }
}

bool
block_state::is(protocol_state state) const
{
#ifdef USE_VM
    vm::Bitmap &bitmap = GetBitmap();
    return bitmap.isAnyInRange(block().acceleratorAddr(GetMode(), block().addr()), block().size(), state);
#else
    for (unsigned i = 0; i < subBlockState_.size(); i++) {
        if (subBlockState_[i] == state) return true;
    }

    return false;
#endif
}

void
block_state::reset()
{
    faultsRead_  = 0;
    faultsWrite_ = 0;

    if (util::params::ParamSubBlockStride) strideInfo_.reset();
    if (util::params::ParamSubBlockTree) treeInfo_.reset();
}

int
block_state::unprotect()
{
    int ret = 0;
    unsigned start = 0;
    unsigned size = 0;
#ifdef USE_VM
    vm::Bitmap &bitmap = GetBitmap();
#endif
    for (unsigned i = 0; i < subBlocks_; i++) {
#ifdef USE_VM
        protocol_state state = bitmap.getEntry<protocol_state>(block().acceleratorAddr(GetMode()) + i * SubBlockSize_);
#else
        protocol_state state = protocol_state(subBlockState_[i]);
#endif
        if (state == lazy_types::Dirty) {
            if (size == 0) start = i;
            size++;
        } else if (size > 0) {
            ret = memory_ops::protect(getSubBlockAddr(start), SubBlockSize_ * size, GMAC_PROT_READWRITE);
            if (ret < 0) break;
            size = 0;
        }
    }
    if (size > 0) {
        ret = memory_ops::protect(getSubBlockAddr(start), SubBlockSize_ * size, GMAC_PROT_READWRITE);
    }
    return ret;
}

int
block_state::protect(GmacProtection prot)
{
    int ret = 0;
#if 1
    ret = memory_ops::protect(block().addr(), block().size(), prot);
#else
    for (unsigned i = 0; i < subBlockState_.size(); i++) {
        if (subBlockState_[i] == lazy_types::Dirty) {
            ret = memory_ops::protect(getSubBlockAddr(i), SubBlockSize_, prot);
            if (ret < 0) break;
        }
    }
#endif
    return ret;
}

void
block_state::acquired()
{
    //setAll(lazy::Invalid);   
    //state_ = lazy::Invalid;
#ifdef DEBUG
    //::memset(&subBlockFaults_[0], 0, subBlockFaults_.size() * sizeof(long_t));
#endif
}

void
block_state::released()
{
    //setAll(lazy::ReadOnly);   
    state_ = lazy_types::ReadOnly;
    reset();
#ifdef DEBUG
    //::memset(&subBlockFaults_[0], 0, subBlockFaults_.size() * sizeof(long_t));
#endif
}

gmacError_t
block_state::dump(std::ostream &stream, common::Statistic stat)
{
#ifdef DEBUG
    if (stat == common::PAGE_FAULTS_READ) {
        for (unsigned i = 0; i < subBlockFaultsRead_.size(); i++) {
            std::ostringstream oss;
            oss << subBlockFaultsRead_[i] << " ";

            stream << oss.str();
        }
        ::memset(&subBlockFaultsRead_[0], 0, subBlockFaultsRead_.size() * sizeof(long_t));
    } else if (stat == common::PAGE_FAULTS_WRITE) {
        for (unsigned i = 0; i < subBlockFaultsWrite_.size(); i++) {
            std::ostringstream oss;
            oss << subBlockFaultsWrite_[i] << " ";

            stream << oss.str();
        }
        ::memset(&subBlockFaultsWrite_[0], 0, subBlockFaultsWrite_.size() * sizeof(long_t));

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

#include "memory/address_space.h"

#include "block.h"

namespace __impl {
namespace memory {
namespace protocols {
namespace lazy_types {
#if 0
block &
block_state::block()
{
	return *(lazy_types::block *)this;
}

const block &
block_state::block() const
{
	return *(const lazy_types::block *)this;
}
#endif

block::block(object &parent, size_t offset, size_t size, State init) :
    common::block(parent, offset, size),
    state_(init)
#ifdef DEBUG
    , faultsRead_(0),
    faultsWrite_(0),
    transfersToAccelerator_(0),
    transfersToHost_(0)
#endif
{
}

State
block::get_state() const
{
    return state_;
}

void
block::set_state(lazy_types::State state, host_ptr /* addr */)
{
    state_ = state;
}

hal::event_ptr
block::sync_to_device(gmacError_t &err)
{
    TRACE(LOCAL, "Transfer block to device: %p", get_bounds().start);

#ifdef DEBUG
    transfersToAccelerator_++;
#endif
    hal::event_ptr ret;
    ret = get_owner()->copy_async(parent::get_device_addr(),
                                  hal::const_ptr(get_shadow()),
                                  size(), err);
    return ret;
}

hal::event_ptr
block::sync_to_host(gmacError_t &err)
{
    TRACE(LOCAL, "Transfer block to host: %p", get_bounds().start);

#ifdef DEBUG
    transfersToHost_++;
#endif
    hal::event_ptr ret;
    err = get_owner()->copy(hal::ptr(get_shadow()),
                            parent::get_device_const_addr(),
                            size());
    return ret;
}

void
block::read(host_const_ptr /*addr*/)
{
#ifdef DEBUG
    faultsRead_++;
#endif
    faultsCacheRead_++;
}

void
block::write(host_const_ptr /*addr*/)
{
#ifdef DEBUG
    faultsWrite_++;
#endif
    faultsCacheWrite_++;
}

bool
block::is(lazy_types::State state) const
{
    return state_ == state;
}

int
block::protect(GmacProtection prot)
{
    return memory_ops::protect(get_bounds().start, size(), prot);
}

int
block::unprotect()
{
    return memory_ops::protect(get_bounds().start, size(), GMAC_PROT_READWRITE);
}

void
block::acquired()
{
#ifdef DEBUG
    faultsRead_ = 0;
    faultsWrite_ = 0;
#endif

    state_ = lazy_types::Invalid;
}

void
block::released()
{
#ifdef DEBUG
    faultsRead_ = 0;
    faultsWrite_ = 0;
#endif

    state_ = lazy_types::ReadOnly;
}

address_space_ptr
block::get_owner() const
{
    return parent_.get_owner();
}

hal::ptr
block::get_device_addr(host_ptr addr)
{
    return parent_.get_device_addr(addr);
}

hal::const_ptr
block::get_device_const_addr(host_const_ptr addr) const
{
    return parent_.get_device_const_addr(addr);
}

gmacError_t
block::dump(std::ostream &stream, common::Statistic stat)
{
#ifdef DEBUG
    if (stat == common::PAGE_FAULTS_READ) {
        std::ostringstream oss;
        oss << faultsRead_ << " ";
        stream << oss.str();

        faultsRead_ = 0;
    } else if (stat == common::PAGE_FAULTS_WRITE) {
        std::ostringstream oss;
        oss << faultsWrite_ << " ";
        stream << oss.str();

        faultsWrite_ = 0;
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

#endif /* BLOCKSTATE_IMPL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
