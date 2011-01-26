/* Copyright (c) 2009, 2011 University of Illinois
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

#ifndef GMAC_MEMORY_BITMAP_H_
#define GMAC_MEMORY_BITMAP_H_

#include <cmath>

#include "config/common.h"

#include "util/Lock.h"
#include "util/Logger.h"


#ifdef USE_VM

#ifdef BITMAP_BYTE
#else
#ifdef BITMAP_BIT
#else
#error "ERROR: Bitmap granularity not defined!"
#endif
#endif


namespace __impl {

namespace core {
class Mode;
}

namespace memory  { namespace vm {

enum BitmapState {
    BITMAP_UNSET    = 0,
    BITMAP_SET_ACC  = 1,
    BITMAP_SET_HOST = 2
};
typedef uint8_t BitmapType;


class GMAC_LOCAL Node
{
private:
    unsigned level_;
    size_t nEntries_;
    size_t nUsedEntries_;
    std::vector<bool> usedEntries_;

    long_t firstUsedEntry_;
    long_t lastUsedEntry_;

protected:
    long_t mask_;
    unsigned shift_;

    std::vector<unsigned> nextEntries_;

    long_t getLocalIndex(long_t index) const;
    long_t getGlobalIndex(long_t localIndex) const;
    long_t getNextIndex(long_t index) const;

    virtual Node *createChild() = 0;

    Node(unsigned level, size_t nEntries, std::vector<unsigned> nextEntries);
public:
    virtual ~Node() {}

    virtual BitmapState getEntry(long_t index) = 0;
    virtual BitmapState getAndSetEntry(long_t index, BitmapState state) = 0;
    virtual void setEntry(long_t index, BitmapState state) = 0;
    virtual void setEntryRange(long_t startIndex, long_t endIndex, BitmapState state) = 0;
    virtual bool isAnyInRange(long_t startIndex, long_t endIndex, BitmapState state) = 0;

    virtual void registerRange(long_t startIndex, long_t endIndex) = 0;
    virtual void unregisterRange(long_t startIndex, long_t endIndex) = 0;
    
    unsigned getLevel() const;
    size_t getNUsedEntries() const;
    void setNUsedEntries(size_t usedEntries) const;

    long_t getFirstUsedEntry() const;
    long_t getLastUsedEntry() const;

    void addEntries(long_t startIndex, long_t endIndex);
    void removeEntries(long_t startIndex, long_t endIndex);
};

class Bitmap;

class GMAC_LOCAL StoreHost
{
protected:
    Bitmap &root_;
    hostptr_t entriesHost_;

    size_t size_;

    StoreHost(Bitmap &root, size_t size, bool alloc = true);
    virtual ~StoreHost();
};

class GMAC_LOCAL StoreShared : public StoreHost
{
private:
    void syncToHost(long_t startIndex, long_t endIndex, size_t elemSize);
    void syncToAccelerator(long_t startIndex, long_t endIndex, size_t elemSize);
    void setDirty(bool synced);

protected:
    hostptr_t entriesAccHost_;
    accptr_t entriesAcc_;

    bool dirty_;
    bool synced_;

    long_t firstDirtyEntry_;
    long_t lastDirtyEntry_;

    bool isDirty() const;

    StoreShared(Bitmap &root, size_t size, bool allocHost = true);
    virtual ~StoreShared();

    void allocAcc(bool isRoot);
    void freeAcc(bool isRoot);

    virtual void acquire() = 0;
    virtual void release() = 0;

    long_t getFirstDirtyEntry();
    long_t getLastDirtyEntry();

    void addDirtyEntry(long_t index);
    void addDirtyEntries(long_t startIndex, long_t endIndex);

public:
    template <typename T>
    void syncToHost(long_t startIndex, long_t endIndex);
    template <typename T>
    void syncToAccelerator(long_t startIndex, long_t endIndex);

    bool isSynced() const;
    void setSynced(bool synced);

    accptr_t getAccAddr() const;
};

template <typename S>
class NodeStore :
    public Node,
    public S
{
protected:
    Node *getNode(long_t index);
    virtual BitmapState getLeaf(long_t index) = 0;
    virtual uint8_t &getLeafRef(long_t index) = 0;

    Node *&getNodeRef(long_t index);

    NodeStore(unsigned level, Bitmap &root, size_t nEntries, std::vector<unsigned> nextEntries);
    virtual ~NodeStore();
public:

    virtual BitmapState getEntry(long_t index);
    virtual BitmapState getAndSetEntry(long_t index, BitmapState state);
    virtual void setEntry(long_t index, BitmapState state);
    virtual void setEntryRange(long_t startIndex, long_t endIndex, BitmapState state);
    virtual bool isAnyInRange(long_t startIndex, long_t endIndex, BitmapState state);
};

#if 0
class GMAC_LOCAL NodeHost : public NodeStore<StoreHost>
{
public:
    NodeHost(Bitmap &root, size_t nEntries, std::vector<unsigned> nextEntries);
    virtual ~NodeHost();

    void registerRange(long_t startIndex, long_t endIndex);

    void acquire() {}
    void release() {}
};

#endif

class GMAC_LOCAL NodeHost : public NodeStore<StoreHost>
{
protected:
    Node *createChild();

    virtual BitmapState getLeaf(long_t index);
    virtual uint8_t &getLeafRef(long_t index);

public:
    NodeHost(unsigned level, Bitmap &root, size_t nEntries, std::vector<unsigned> nextEntries);

    void registerRange(long_t startIndex, long_t endIndex);
    void unregisterRange(long_t startIndex, long_t endIndex);
};

class GMAC_LOCAL NodeShared : public NodeStore<StoreShared>
{
    friend class BitmapShared;

    void sync();

protected:
    Node *createChild();
    NodeShared *&getNodeRef(long_t index);
    NodeShared *&getNodeAccHostRef(long_t index);
    NodeShared *getNodeAccAddr(long_t index);

    BitmapState getLeaf(long_t index);
    uint8_t &getLeafRef(long_t index);

public:
    NodeShared(unsigned level, Bitmap &root, size_t nEntries, std::vector<unsigned> nextEntries);
    ~NodeShared();

    BitmapState getEntry(long_t index);
    BitmapState getAndSetEntry(long_t index, BitmapState state);
    void setEntry(long_t index, BitmapState state);
    void setEntryRange(long_t startIndex, long_t endIndex, BitmapState state);
    bool isAnyInRange(long_t startIndex, long_t endIndex, BitmapState state);

    void registerRange(long_t startIndex, long_t endIndex);
    void unregisterRange(long_t startIndex, long_t endIndex);

    void acquire();
    void release();
};

class GMAC_LOCAL Bitmap
{
    friend class StoreShared;

protected:
    /**
     * Number of levels of the bitmap
     */
    static const unsigned &BitmapLevels_;

    /**
     * Number of entries in the first level of the bitmap
     */
    static const unsigned &L1Entries_;

    /**
     * Number of entries in the second level of the bitmap
     */
    static const unsigned &L2Entries_;

    /**
     * Number of entries in the third level of the bitmap
     */
    static const unsigned &L3Entries_;

    static long_t L1Mask_;
    static long_t L2Mask_;
    static long_t L3Mask_;
    static unsigned L1Shift_;
    static unsigned L2Shift_;
    static unsigned L3Shift_;

    /**
     * Size in bytes of a block
     */
    static const size_t &BlockSize_;

    /**
     * Number of subblocks per block
     */
    static const unsigned &SubBlocks_;

    /**
     * Mode whose memory is managed by the bitmap
     */
    core::Mode &mode_;

    /**
     * Pointer to the first level (root level) of the bitmap
     */
    Node *root_;

    /**
     * Map of the registered memory ranges
     */
    std::map<accptr_t, size_t> ranges_;

public:
    /**
     * Constructs a new Bitmap
     */
    Bitmap(core::Mode &mode, bool shared);
    ~Bitmap();

    static void Init();

    void cleanUp();

    /**
     * Sets the state of the subblock containing the given address
     *
     * \param addr Address of the subblock to be set
     * \param state State to be set to the block
     */
    void setEntry(const accptr_t addr, BitmapState state);

    /**
     * Sets the state of the subblocks within the given range
     *
     * \param addr Initial address of the subblocks to be set
     * \param bytes Size in bytes of the range
     * \param state State to be set to the blocks
     */
    void setEntryRange(const accptr_t addr, size_t bytes, BitmapState state);

    long_t getIndex(const accptr_t ptr) const;

    /**
     * Gets the state of the subblock containing the given address
     *
     * \param addr Address of the subblock to retrieve the state from
     * \return The state of the subblock
     */
    BitmapState getEntry(const accptr_t addr) const;

    BitmapState getAndSetEntry(const accptr_t addr, BitmapState state);

    bool isAnyInRange(const accptr_t addr, size_t size, BitmapState state);

    /**
     * Registers the given memory range to be managed by the Bitmap
     *
     * \param addr Initial address of the memory range
     * \param bytes Size in bytes of the memory range
     */
    void registerRange(const accptr_t addr, size_t bytes);

    /**
     * Unegisters the given memory range from the Bitmap
     *
     * \param addr Initial address of the memory range
     * \param bytes Size in bytes of the memory range
     */
    void unregisterRange(const accptr_t addr, size_t bytes);
};

class GMAC_LOCAL BitmapHost :
    public Bitmap
{
public:
    BitmapHost(core::Mode &mode);
};

class GMAC_LOCAL BitmapShared :
    public Bitmap
{
protected:
    bool released_;

    void syncToAccelerator();
public:
    BitmapShared(core::Mode &mode);

    void acquire();
    void release();

    bool isReleased() const;
};

enum ModelDirection {
    MODEL_TOHOST = 0,
    MODEL_TODEVICE = 1
};

template <ModelDirection M>
static inline
float costTransferCache(const size_t subBlockSize, size_t subBlocks)
{
    if (M == MODEL_TOHOST) {
        if (subBlocks * subBlockSize <= util::params::ParamModelL1/2) {
            return util::params::ParamModelToHostTransferL1;
        } else if (subBlocks * subBlockSize <= util::params::ParamModelL2/2) {
            return util::params::ParamModelToHostTransferL2;
        } else {
            return util::params::ParamModelToHostTransferMem;
        }
    } else {
        if (subBlocks * subBlockSize <= util::params::ParamModelL1/2) {
            return util::params::ParamModelToDeviceTransferL1;
        } else if (subBlocks * subBlockSize <= util::params::ParamModelL2/2) {
            return util::params::ParamModelToDeviceTransferL2;
        } else {
            return util::params::ParamModelToDeviceTransferMem;
        }
    }
}

template <ModelDirection M>
static inline
float costGaps(const size_t subBlockSize, unsigned gaps, unsigned subBlocks)
{
    return costTransferCache<M>(subBlockSize, subBlocks) * gaps * subBlockSize;
}

template <ModelDirection M>
static inline
float costTransfer(const size_t subBlockSize, size_t subBlocks)
{
    return costTransferCache<M>(subBlockSize, subBlocks) * subBlocks * subBlockSize;
}

template <ModelDirection M>
static inline
float costConfig()
{
    if (M == MODEL_TOHOST) {
        return util::params::ParamModelToHostConfig;
    } else {
        return util::params::ParamModelToDeviceConfig;
    }
}

template <ModelDirection M>
static inline
float cost(const size_t subBlockSize, size_t subBlocks)
{
    return costConfig<M>() + costTransfer<M>(subBlockSize, subBlocks);
}

}}}

#include "Bitmap-impl.h"


#endif
#endif
