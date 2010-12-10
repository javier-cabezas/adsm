/* Copyright (c) 2009 University of Illinois
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

class GMAC_LOCAL Bitmap :
    public __impl::util::Logger,
    protected gmac::util::RWLock {
private:
    core::Mode &mode_;
    typedef uint8_t T;
    hostptr_t bitmap_;

    bool dirty_;
    bool synced_;

    size_t subBlockSize_;
    unsigned subBlockMask_;
    unsigned pageMask_;

    accptr_t accelerator_;

    static const unsigned entriesPerByte;
    unsigned shiftBlock_;
    unsigned shiftPage_;
#ifdef BITMAP_BIT
    uint32_t bitMask_;
#endif
    size_t size_;

    void allocate();

    template <bool check, bool clear, bool set>
    bool CheckClearSet(const accptr_t addr);
    int minEntry_;
    int maxEntry_;

    void updateMaxMin(unsigned entry);

    uint32_t offset(const accptr_t addr) const;

    bool checkAndSet(const accptr_t addr);
    void clear(const accptr_t addr);

public:
    Bitmap(core::Mode &mode, unsigned bits = 32);
    virtual ~Bitmap();

    void cleanUp();

    accptr_t accelerator();
    hostptr_t host() const;

    bool check(const accptr_t addr);
    bool checkBlock(const accptr_t addr);
    void set(const accptr_t addr);
    void setBlock(const accptr_t addr);
    bool checkAndClear(const accptr_t addr);

    const size_t size() const;

    void newRange(const accptr_t ptr, size_t count);
    void removeRange(const accptr_t ptr, size_t count);

    bool clean() const;

    void syncHost();
    void syncAccelerator();
    void reset();

#ifdef DEBUG_BITMAP
    void dump();
#endif

    unsigned getSubBlock(const accptr_t addr) const;
    size_t getSubBlockSize() const;

    bool synced() const;
    void synced(bool s);
};

}}}

#include "Bitmap-impl.h"

enum ModelDirection {
    MODEL_TOHOST = 0,
    MODEL_TODEVICE = 1
};

template <ModelDirection M>
static inline
float costTransferCache(const size_t subBlockSize, size_t subBlocks)
{
    if (M == MODEL_TOHOST) {
        if (subBlocks * subBlockSize <= paramModelL1/2) {
            return paramModelToHostTransferL1;
        } else if (subBlocks * subBlockSize <= paramModelL2/2) {
            return paramModelToHostTransferL2;
        } else {
            return paramModelToHostTransferMem;
        }
    } else {
        if (subBlocks * subBlockSize <= paramModelL1/2) {
            return paramModelToDeviceTransferL1;
        } else if (subBlocks * subBlockSize <= paramModelL2/2) {
            return paramModelToDeviceTransferL2;
        } else {
            return paramModelToDeviceTransferMem;
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
        return paramModelToHostConfig;
    } else {
        return paramModelToDeviceConfig;
    }
}

template <ModelDirection M>
static inline
float cost(const size_t subBlockSize, size_t subBlocks)
{
    return costConfig<M>() + costTransfer<M>(subBlockSize, subBlocks);
}

#endif

#endif
