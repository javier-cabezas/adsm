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
	class ResourceManager;
}

namespace memory {

template<typename State>
class GMAC_LOCAL BlockGroup :
    util::gmac_base<BlockGroup<State> >,
    public memory::object {
protected:
    hostptr_t shadow_;
    bool hasUserMemory_;
#if 0
    typedef std::map<accptr_t, std::list<core::address_space *> > AcceleratorMap;
    typedef std::map<core::address_space *, accptr_t> aspace_map;

    AcceleratorMap acceleratorAddr_;
    aspace_map owners_;
#endif
    accptr_t deviceAddr_;
    util::smart_ptr<core::address_space>::shared ownerShortcut_;

    gmacError_t repopulateBlocks(core::address_space &aspace);

    void modifiedObject();
public:
    BlockGroup(Protocol &protocol, hostptr_t cpuAddr, size_t size, typename State::ProtocolState init, gmacError_t &err);
    virtual ~BlockGroup();

    accptr_t get_device_addr(const hostptr_t addr) const;
    accptr_t get_device_addr() const;

    core::address_space &owner();
    const core::address_space &owner() const;

    gmacError_t addOwner(util::smart_ptr<core::address_space>::shared owner);
    gmacError_t removeOwner(util::smart_ptr<core::address_space>::shared owner);

    gmacError_t mapToAccelerator();
    gmacError_t unmapFromAccelerator();

    static gmacError_t split(BlockGroup &group, size_t offset, size_t size);
};

}}

#include "BlockGroup-impl.h"

#endif
