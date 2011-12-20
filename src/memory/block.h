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

#ifndef GMAC_MEMORY_BLOCK_H_
#define GMAC_MEMORY_BLOCK_H_

#include "config/common.h"
#include "config/config.h"

#include "include/gmac/types.h"
#include "memory/protocol.h"
#include "trace/logger.h"
#include "util/lock.h"
#include "util/misc.h"

#include "util/smart_ptr.h"
#if 0
/** Description for __impl. */
namespace __impl {

namespace core {
    class address_space;

    typedef util::shared_ptr<address_space> address_space_ptr;
    typedef util::shared_ptr<const address_space> address_space_const_ptr;
}

namespace memory {

class block;

typedef __impl::util::shared_ptr<block> block_ptr;

/** Memory block
 * A memory block is a coherence unit of shared memory objects in GMAC, which are a collection of memory blocks.  Each
 * memory block has an unique host memory address, used by applications to access the shared data in the CPU code, and
 * a shadow host memory address used by GMAC to update the contents of the block. Upon creation, a memory block also
 * has one or more accelerator memory addresses, used by the application to access the data from the accelerator, and
 * owners which are those execution modes allowed to access the memory block from the accelerator. However, a memory
 * block might lose its owner and memory addresses (e.g., when the execution mode owning the memory block dies) and
 * stil be accessible from the CPU.
 * Memory block methods should only be called from GMAC objects and GMAC memory coherence protocols.
 */
class GMAC_LOCAL block :
	public gmac::util::mutex<block> {
    DBC_FORCE_TEST(block)

	typedef gmac::util::mutex<block> Lock;

protected:
    /** Block size (in bytes) */
    size_t size_;

    /** Host address where for applications to access the block. */
    hostptr_t addr_;

    /** Shadow host memory mapping that is always read/write. */
    hostptr_t shadow_;

    /**
     * Default construcutor
     *
     * \param addr Host memory address for applications to accesss the block
     * \param shadow Shadow host memory mapping that is always read/write
     * \param size Size (in bytes) of the memory block
     */
    block(hostptr_t addr, hostptr_t shadow, size_t size);

public:
    typedef util::bounds<hostptr_t> bounds;

public:
    /**
     * Get memory block owner
     * \return Owner of the memory block
     */
    virtual core::address_space_ptr get_owner() const = 0;

    /**
     * Get memory block address at the accelerator
     * \return Accelerator memory address of the block
     */
    virtual accptr_t get_device_addr(const hostptr_t addr) const = 0;

    /**
     * Get memory block address at the accelerator
     * \return Accelerator memory address of the block
     */
    virtual accptr_t get_device_addr() const = 0;

    /**
     * Dump statistics about the memory block
     * \param param Stream to dump the statistics to
     * \param stat Statistic to be dumped
     * \return Error code
     */
    gmacError_t dump(std::ostream &param, protocols::common::Statistic stat);

    hostptr_t get_shadow() const;
};

}}

#include "block-impl.h"

#ifdef USE_DBC
//#include "memory/dbc/block.h"
#endif

#endif

#endif
