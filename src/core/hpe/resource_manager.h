/* Copyright (c) 2009-2011sity of Illinois
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

#ifndef GMAC_CORE_HPE_RESOURCE_MANAGER_H_
#define GMAC_CORE_HPE_RESOURCE_MANAGER_H_

#include <map>
#include <vector>

#include "config/common.h"
#include "config/order.h"

#include "hal/types.h"

#include "util/stl/locked_map.h"

namespace __impl { namespace core { namespace hpe {

class address_space;
class context;
class process;
class thread;
class vdevice;

struct GMAC_LOCAL address_space_resources {
    hal::context_t *context_;

    hal::stream_t *streamLaunch_;
    hal::stream_t *streamToAccelerator_;
    hal::stream_t *streamToHost_;
    hal::stream_t *streamAccelerator_;
};

typedef util::shared_ptr<address_space> address_space_ptr;

/** Represents the resources used by a running process */
class GMAC_LOCAL resource_manager {
    // Needed to let Singleton call the protected constructor
    friend class process;
protected:
    process &proc_;

    typedef util::stl::locked_map<GmacAddressSpaceId, address_space_ptr> map_aspace;
    typedef util::stl::locked_map<address_space *, address_space_resources> map_aspace_resources;

    map_aspace aspaceMap_;
    map_aspace_resources aspaceResourcesMap_;

    std::vector<hal::device *> devices_;

    /**
     * Constructs the resource manager
     */
    resource_manager(process &proc);
    /**
     * Destroys the process and releases the resources used by it
     */
    virtual ~resource_manager();

    gmacError_t init_thread(thread &t, const thread *parent);

public:
    gmacError_t register_device(hal::device &dev);

    address_space_ptr create_address_space(unsigned deviceId, gmacError_t &err);
    gmacError_t destroy_address_space(address_space &aspace);

    address_space_ptr get_address_space(GmacAddressSpaceId aspaceId);

    vdevice *create_virtual_device(GmacAddressSpaceId id, gmacError_t &err);
    gmacError_t destroy_virtual_device(vdevice &dev);

    //context *create_context(THREAD_T id, address_space &aspace);

    unsigned get_number_of_devices() const;
    bool are_all_devices_integrated() const;
    gmacError_t get_device_info(unsigned deviceId, GmacDeviceInfo &info);
    gmacError_t get_device_free_mem(unsigned deviceId, size_t &freeMem);
};

}}}

#endif
