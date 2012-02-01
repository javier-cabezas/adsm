/* Copyright (c) 2009, 2010 University of Illinois
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

#ifndef GMAC_MEMORY_DBC_MANAGER_H_
#define GMAC_MEMORY_DBC_MANAGER_H_

#include "config/dbc/types.h"
#include "config/dbc/Contract.h"
#include "memory/manager.h"

namespace __dbc { namespace memory {

//! Memory Manager Interface

//! Memory Managers implement a policy to move data from/to
//! the CPU memory to/from the accelerator memory.
class GMAC_LOCAL manager :
    public __impl::memory::manager,
    public virtual Contract {
    DBC_TESTED(__impl::memory::manager)

private:
    typedef __impl::hal::device_input device_input_impl;
    typedef __impl::hal::device_output device_output_impl;

    typedef __impl::memory::address_space_ptr address_space_ptr_impl;
    typedef __impl::memory::list_addr list_addr_impl;
    typedef __impl::memory::manager parent;

protected:

    /**
     * Default destructor
     */
    ~manager();
public:
    /**
     * Default constructor
     */
    manager();

    size_t get_alloc_size(address_space_ptr_impl aspace, host_const_ptr addr, gmacError_t &err) const;

    gmacError_t alloc(address_space_ptr_impl aspace, host_ptr *addr, size_t size);
    gmacError_t free(address_space_ptr_impl aspace, host_ptr addr);

    address_space_ptr_impl get_owner(host_const_ptr addr, size_t size = 0);

    gmacError_t acquire_objects(address_space_ptr_impl aspace, const list_addr_impl &addrs = __impl::memory::AllAddresses);
    gmacError_t release_objects(address_space_ptr_impl aspace, const list_addr_impl &addrs = __impl::memory::AllAddresses);

    bool signal_read(address_space_ptr_impl aspace, host_ptr addr);
    bool signal_write(address_space_ptr_impl aspace, host_ptr addr);

    gmacError_t from_io_device(address_space_ptr_impl aspace, host_ptr addr, device_input_impl &input, size_t count);
    gmacError_t to_io_device(device_output_impl &output, address_space_ptr_impl aspace, host_const_ptr addr, size_t count);

    gmacError_t memcpy(address_space_ptr_impl aspace, host_ptr dst, host_const_ptr src, size_t size);
    gmacError_t memset(address_space_ptr_impl aspace, host_ptr dst, int c, size_t size);

    gmacError_t flush_dirty(address_space_ptr_impl aspace);
};

}}

#endif
