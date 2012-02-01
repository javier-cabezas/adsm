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
 WITH THE SOFTWARE.
 */

#ifndef GMAC_CORE_HPE_ADDRESS_SPACE_HOST_H_
#define GMAC_CORE_HPE_ADDRESS_SPACE_HOST_H_

#include "address_space.h"

namespace __impl { namespace core { namespace hpe {

class GMAC_LOCAL address_space_host :
    public memory::address_space,
    private gmac::util::mutex<address_space> {

public:
    address_space_host(process &proc);

    virtual ~address_space_host();

    gmacError_t map(hal::ptr &dst, host_ptr src, size_t count, unsigned align = 1);
    gmacError_t unmap(host_ptr addr, size_t count);

    hal::ptr alloc_host_pinned(size_t count, gmacError_t &err);
    gmacError_t free_host_pinned(hal::ptr ptr);

    hal::ptr get_host_pinned_mapping(host_ptr ptr, gmacError_t &err);

    gmacError_t copy(hal::ptr dst, hal::const_ptr src, size_t count);
    gmacError_t copy(hal::ptr dst, hal::device_input &input, size_t count);
    gmacError_t copy(hal::device_output &output, hal::const_ptr src, size_t count);

    hal::event_ptr copy_async(hal::ptr dst, hal::const_ptr src, size_t count, gmacError_t &err);
    hal::event_ptr copy_async(hal::ptr dst, hal::device_input &input, size_t count, gmacError_t &err);
    hal::event_ptr copy_async(hal::device_output &output, hal::const_ptr src, size_t count, gmacError_t &err);

    gmacError_t memset(hal::ptr addr, int c, size_t count)
    {
        ASSERTION(addr.is_host_ptr());

        ::memset(addr.get_host_addr(), c, count);
        return gmacSuccess;
    }

    hal::event_ptr memset_async(hal::ptr addr, int c, size_t count, gmacError_t &err)
    {
        ASSERTION(addr.is_host_ptr());

        ::memset(addr.get_host_addr(), c, count);
        err = gmacSuccess;
        return hal::event_ptr();
    }

    bool is_integrated() const
    {
        return true;
    }

    bool has_direct_copy(memory::address_space_const_ptr aspace) const;
};

}}}

#endif
