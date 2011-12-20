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

#ifndef GMAC_MEMORY_PROTOCOL_LAZY_BLOCKSTATE_H_
#define GMAC_MEMORY_PROTOCOL_LAZY_BLOCKSTATE_H_

#include "memory/block.h"
#include "util/ReusableObject.h"

#include "memory/protocol/common/block_state.h"

namespace __impl {
namespace memory {
namespace protocols {
namespace lazy_types {

//! Protocol states
enum State {
    ReadOnly = 0, /*!< Valid copy of the data in both host and accelerator memory */
    Invalid  = 1, /*!< Valid copy of the data in accelerator memory */
    Dirty    = 2, /*!< Valid copy of the data in host memory */
    HostOnly = 3  /*!< Data only allowed in host memory */
};

#if defined(USE_SUBBLOCK_TRACKING) || defined(USE_VM)
template <typename T>
class Array {
    T *array_;
    size_t size_;
public:
    explicit Array(size_t size) :
        size_(size)
    {
        array_ = new T[size];
    }

    ~Array()
    {
        delete [] array_;
    }

    T &operator[](const unsigned index)
    {
        ASSERTION(index < size_, "Index: %u. Size: "FMT_SIZE, index, size_);
        return array_[index];
    }

    const T &operator[](const unsigned index) const
    {
        ASSERTION(index < size_, "Index: %u. Size: "FMT_SIZE, index, size_);
        return array_[index];
    }

    size_t size() const
    {
        return size_;
    }
};

typedef Array<uint8_t> SubBlocks;
typedef Array<long_t> SubBlockCounters;

/// Tree used to group subblocks and speculatively unprotect them
struct GMAC_LOCAL BlockTreeState : public util::reusable<BlockTreeState> {
    unsigned counter_;
    BlockTreeState();
};

class GMAC_LOCAL StrideInfo {
protected:
    block &block_;

    unsigned stridedFaults_;
    long_t stride_;
    hostptr_t lastAddr_;
    hostptr_t firstAddr_;

public:
    StrideInfo(block &block);

    void signal_write(hostptr_t addr);

    unsigned getStridedFaults() const;
    bool isStrided() const;
    hostptr_t getFirstAddr() const;
    long_t getStride() const;

    void reset();
};


class GMAC_LOCAL BlockTreeInfo {
public:
    typedef std::pair<unsigned, unsigned> Pair;
protected:
    block &block_;

    unsigned treeStateLevels_;
    BlockTreeState *treeState_;

    Pair lastUnprotectInfo_;

    Pair increment(unsigned subBlock);
public:
    BlockTreeInfo(block &block);
    ~BlockTreeInfo();

    void signal_write(const hostptr_t addr);
    Pair getUnprotectInfo();

    void reset();
};

#endif

class GMAC_LOCAL block :
    public common::block_state {
#if defined(USE_SUBBLOCK_TRACKING)
    friend class StrideInfo;
    friend class BlockTreeInfo;
#endif

protected:
    State state_;

#if defined(USE_SUBBLOCK_TRACKING)
    //const lazy::Block &block();
    unsigned subBlocks_;
    SubBlocks subBlockState_; 
#endif

#ifdef DEBUG
    // Global statistis
#if defined(USE_SUBBLOCK_TRACKING)
    SubBlockCounters subBlockFaultsRead_; 
    SubBlockCounters subBlockFaultsWrite_; 
    SubBlockCounters transfersToAccelerator_; 
    SubBlockCounters transfersToHost_; 
#else
    unsigned faultsRead_;
    unsigned faultsWrite_;
    unsigned transfersToAccelerator_;
    unsigned transfersToHost_;
#endif // USE_SUBBLOCK_TRACKING
#endif

#if defined(USE_SUBBLOCK_TRACKING)
    // Speculative subblock unprotect policies
    StrideInfo strideInfo_;
    BlockTreeInfo treeInfo_;

    void setSubBlock(const hostptr_t addr, protocol_state state);
    void setSubBlock(long_t subBlock, protocol_state state);
    void setAll(protocol_state state);

    void reset();

    hostptr_t getSubBlockAddr(const hostptr_t addr) const;
    hostptr_t getSubBlockAddr(unsigned index) const;
    unsigned getSubBlocks() const;
    size_t getSubBlockSize() const;

    void writeStride(const hostptr_t addr);
    void writeTree(const hostptr_t addr);
#endif

public:
    block(object &parent, hostptr_t addr, hostptr_t shadow, size_t size, State init);

    State get_state() const;
    void set_state(State state, hostptr_t addr = NULL);

#if 0
    bool hasState(protocol_state state) const;
#endif

    hal::event_ptr sync_to_device(gmacError_t &err);
    hal::event_ptr sync_to_host(gmacError_t &err);     

    void read(const hostptr_t addr);
    void write(const hostptr_t addr);

    bool is(State state) const;

    int protect(GmacProtection prot);
    int unprotect();

    void acquired();
    void released();

    /**
     * Get memory block owner
     *
     * \return A reference to the owner mode of the memory block
     */
    core::address_space_ptr get_owner() const;

    /**
     * Get memory block address at the accelerator
     *
     * \param current Execution mode requesting the operation
     * \param addr Address within the block
     * \return Accelerator memory address of the block
     */
    accptr_t get_device_addr(const hostptr_t addr) const;

    /**
     * Get memory block address at the accelerator
     *
     * \return Accelerator memory address of the block
     */
    accptr_t get_device_addr() const;

    gmacError_t dump(std::ostream &stream, common::Statistic stat);
};

}}}}

#include "block_state-impl.h"

#endif // GMAC_MEMORY_PROTOCOL_LAZY_BLOCKSTATE_H_

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
