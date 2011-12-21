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

#ifndef GMAC_MEMORY_BLOCKGROUP_H_
#define GMAC_MEMORY_BLOCKGROUP_H_

#include "memory/object.h"

#include "util/gmac_base.h"

namespace __impl { 

namespace core {
	class address_space;

	typedef util::shared_ptr<core::address_space> address_space_ptr;
	typedef util::shared_ptr<const core::address_space> address_space_const_ptr;
}

namespace memory {

template <typename ProtocolTraits>
class GMAC_LOCAL object_state :
    util::gmac_base<object_state<ProtocolTraits> >,
    public memory::object {
protected:
    host_ptr shadow_;
    bool hasUserMemory_;

    hal::ptr_t deviceAddr_;
    core::address_space_ptr ownerShortcut_;

    gmacError_t repopulate_blocks(core::address_space &aspace);

    void modified_object();
public:
    object_state(protocol &protocol, host_ptr cpuAddr, size_t size, typename ProtocolTraits::State init, gmacError_t &err);
    virtual ~object_state();

    hal::ptr_t get_device_addr(host_ptr addr);
    hal::ptr_t get_device_addr();

    hal::ptr_const_t get_device_const_addr(host_const_ptr addr) const;
    hal::ptr_const_t get_device_const_addr() const;

    core::address_space_ptr get_owner();
    core::address_space_const_ptr get_owner() const;

    gmacError_t add_owner(core::address_space_ptr owner);
    gmacError_t remove_owner(core::address_space_const_ptr owner);

    gmacError_t map_to_device();
    gmacError_t unmap_from_device();

    static gmacError_t split(object_state &group, size_t offset, size_t size);
};

}}

#include "object_state-impl.h"

#endif
