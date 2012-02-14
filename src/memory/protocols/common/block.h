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

#ifndef GMAC_MEMORY_PROTOCOL_COMMON_BLOCK_H_
#define GMAC_MEMORY_PROTOCOL_COMMON_BLOCK_H_

#include <ostream>

#include "config/common.h"

#include "hal/types.h"

#include "util/misc.h"

#include "memory/protocols/common/types.h"


namespace __impl { namespace memory {

class address_space;
class object;

typedef util::shared_ptr<address_space> address_space_ptr;
typedef util::shared_ptr<const address_space> address_space_const_ptr;

namespace protocols { namespace common {

class GMAC_LOCAL block :
    public gmac::util::mutex<block>,
    public util::unique<block> {
    typedef gmac::util::mutex<block> Lock;
protected:
    /** Object that contains the block */
    object &parent_;

    /** Block size (in bytes) */
    size_t size_;

    /** Host address where for applications to access the block. */
    size_t offset_;

    unsigned faultsCacheWrite_;
    unsigned faultsCacheRead_;

    /**
     * Default constructor
     *
     * \param parent Object containing the block
     * \param addr Host memory address for applications to accesss the block
     * \param shadow Shadow host memory mapping that is always read/write
     * \param size Size (in bytes) of the memory block
     */
    block(object &parent, size_t offset, size_t size);

public:
    typedef util::bounds<host_ptr> bounds;

    /**
     * Host memory address bounds of the block
     * \return The host memory address bounds of the block
     */
    const bounds get_bounds() const;

    /**
     *  Block size
     * \return Size in bytes of the memory block
     */
    size_t size() const;

    /**
     * Get memory block owner
     * \return Owner of the memory block
     */
    virtual address_space_ptr get_owner() const = 0;

    /**
     * Get memory block address at the accelerator
     * \return Accelerator memory address of the block
     */
    virtual hal::ptr get_device_addr(host_ptr addr) = 0;

    /**
     * Get memory block address at the accelerator
     * \return Accelerator memory address of the block
     */
    inline
    hal::ptr get_device_addr()
    {
        return get_device_addr(get_bounds().start);
    }

    /**
     * Get memory block address at the accelerator
     * \return Accelerator memory address of the block
     */
    virtual hal::const_ptr get_device_const_addr(host_const_ptr addr) const = 0;

    /**
     * Get memory block address at the accelerator
     * \return Accelerator memory address of the block
     */
    inline
    hal::const_ptr get_device_const_addr() const
    {
        return get_device_const_addr(get_bounds().start);
    }

    hal::ptr get_shadow() const;

    virtual hal::event_ptr sync_to_device(gmacError_t &err) = 0;
    virtual hal::event_ptr sync_to_host(gmacError_t &err) = 0;

    unsigned get_faults_cache_write() const;
    unsigned get_faults_cache_read() const;

    void reset_faults_cache_write();
    void reset_faults_cache_read();

    /**
     * Dump statistics about the memory block
     * \param param Stream to dump the statistics to
     * \param stat Statistic to be dumped
     * \return Error code
     */
    virtual gmacError_t dump(std::ostream &stream, Statistic stat) = 0;
};

typedef util::shared_ptr<block> block_ptr;
typedef util::shared_ptr<const block> block_const_ptr;

}}}}

#include "block-impl.h"

#endif /* BLOCKINFO_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
