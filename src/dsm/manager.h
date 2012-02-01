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

#ifndef GMAC_DSM_MANAGER_H_
#define GMAC_DSM_MANAGER_H_

#include "config/common.h"
#include "include/gmac/types.h"
#include "util/singleton.h"
#include "util/stl/locked_map.h"

#include "memory/address_space.h"

#include "mapping.h"

namespace __impl {

namespace dsm {

//! DSM Manager Interface

//! DSM Managers orchestate the data transfers between host and accelerator memories
class GMAC_LOCAL manager :
    public __impl::util::singleton<dsm::manager> {
protected:

    typedef std::map<hal::ptr::address, mapping_ptr> map_mappings;

    const hal::context_t::attribute_id AttributeMappings_;

    static void aspace_created(manager *m, hal::context_t &aspace);
    static void aspace_destroyed(manager *m, hal::context_t &aspace);

    map_mappings &get_aspace_mappings(hal::context_t &ctx);

    typedef util::range<map_mappings::iterator> range_mapping;
    range_mapping get_mappings_in_range(map_mappings &mappings, hal::ptr::address begin, size_t count);

    /**
     * Default destructor
     */
    virtual ~manager();
public:
    /**
     * Default constructor
     */
    manager();

    /**
     * Map the given host memory pointer to the accelerator memory. If the given
     * pointer is NULL, host memory is alllocated too.
     * \param mode Execution mode where to allocate memory
     * \param addr Memory address to be mapped or NULL if host memory is requested
     * too
     * \param size Size (in bytes) of shared memory to be mapped 
     * \param flags 
     * \param err Reference to store the error code for the operation
     * \return Address that identifies the allocated memory
     */
    gmacError_t link(hal::ptr ptr1, hal::ptr ptr2, size_t count, int flags);
    gmacError_t unlink(hal::ptr mapping, size_t count);

    gmacError_t acquire(hal::ptr mapping, size_t count, int flags);
    gmacError_t release(hal::ptr mapping, size_t count);

    gmacError_t sync(hal::ptr mapping, size_t count);

    gmacError_t memcpy(hal::ptr dst, hal::ptr src, size_t count);
    gmacError_t memset(hal::ptr ptr, int c, size_t count);

    gmacError_t from_io_device(hal::ptr addr, hal::device_input &input, size_t count);
    gmacError_t to_io_device(hal::device_output &output, hal::const_ptr addr, size_t count);

    gmacError_t flush_dirty(address_space_ptr aspace);
};

}}

#include "manager-impl.h"

#ifdef USE_DBC
#include "memory/dbc/manager.h"
#endif

#endif
