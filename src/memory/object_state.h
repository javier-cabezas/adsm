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

template <typename T>
class block_state;

template<typename State>
class GMAC_LOCAL object_state :
    util::gmac_base<object_state<State> >,
    public memory::object {
protected:
    hostptr_t shadow_;
    bool hasUserMemory_;
#if 0
    typedef std::map<accptr_t, std::list<core::address_space *> > AcceleratorMap;
    typedef std::map<core::address_space *, accptr_t> map_aspace;

    AcceleratorMap acceleratorAddr_;
    map_aspace owners_;
#endif
    accptr_t deviceAddr_;
    core::address_space_ptr ownerShortcut_;

    gmacError_t repopulateBlocks(core::address_space &aspace);

    void modifiedObject();
public:
    object_state(protocol &protocol, hostptr_t cpuAddr, size_t size, typename State::ProtocolState init, gmacError_t &err);
    virtual ~object_state();

    accptr_t get_device_addr(const hostptr_t addr) const;
    accptr_t get_device_addr() const;

    core::address_space_ptr owner();
    core::address_space_const_ptr owner() const;

    gmacError_t addOwner(core::address_space_ptr owner);
    gmacError_t removeOwner(core::address_space_const_ptr owner);

    gmacError_t mapToAccelerator();
    gmacError_t unmapFromAccelerator();

    static gmacError_t split(object_state &group, size_t offset, size_t size);
};

}}

#include "object_state-impl.h"

#endif
