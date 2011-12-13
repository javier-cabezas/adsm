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

#ifndef GMAC_MEMORY_PROTOCOL_DBC_LAZYBASE_H_
#define GMAC_MEMORY_PROTOCOL_DBC_LAZYBASE_H_

#include "config/dbc/types.h"

namespace __dbc { namespace memory { namespace protocols {

class GMAC_LOCAL lazy_base :
    public __impl::memory::protocols::lazy_base,
    public virtual Contract {
    DBC_TESTED(__impl::memory::protocols::lazy_base)

protected:
    lazy_base(bool eager);
    virtual ~lazy_base();

    typedef __impl::memory::protocols::lazy_base parent;
    typedef __impl::memory::block_ptr block_ptr_impl;
    typedef __impl::memory::object object_impl;
    typedef __impl::memory::protocols::lazy_types::State state_impl;
    typedef __impl::memory::protocols::lazy_types::Block lazy_block_impl;
    typedef __impl::util::shared_ptr<lazy_block_impl> lazy_block_ptr_impl;

public:
    __impl::hal::event_ptr signal_read(block_ptr_impl block, hostptr_t addr, gmacError_t &err);
    __impl::hal::event_ptr signal_write(block_ptr_impl block, hostptr_t addr, gmacError_t &err);

    __impl::hal::event_ptr acquire(block_ptr_impl obj, GmacProtection &prot, gmacError_t &err);
    __impl::hal::event_ptr release(block_ptr_impl block, gmacError_t &err);

    __impl::hal::event_ptr release_all(gmacError_t &err);

    __impl::hal::event_ptr map_to_device(block_ptr_impl block, gmacError_t &err);

    __impl::hal::event_ptr unmap_from_device(block_ptr_impl block, gmacError_t &err);

    __impl::hal::event_ptr remove_block(block_ptr_impl block, gmacError_t &err);

    __impl::hal::event_ptr to_host(block_ptr_impl block, gmacError_t &err);

    __impl::hal::event_ptr memset(block_ptr_impl block, size_t blockOffset, int v, size_t size,
                                gmacError_t &err);

    __impl::hal::event_ptr flush_dirty(gmacError_t &err);

    __impl::hal::event_ptr copy_block_to_block(block_ptr_impl d, size_t dstOffset, block_ptr_impl s, size_t srcOffset, size_t count, gmacError_t &err);
};

}}}

#endif
