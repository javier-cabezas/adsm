/* Copyright (c) 2009-2011 University of Illinoisllinois
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

#ifndef GMAC_CORE_HPE_ADDRESS_SPACE_H_
#define GMAC_CORE_HPE_ADDRESS_SPACE_H_

#include <map>

#include "core/address_space.h"
#include "config/common.h"

#include "hal/types.h"

#include "util/lock.h"
#include "util/stl/locked_map.h"

namespace __impl {
    
namespace core {

namespace hpe {

class address_space;
class context;
class kernel;
class process;
class vdevice;

class address_space;

typedef util::shared_ptr<address_space> address_space_ptr;
typedef util::shared_ptr<const address_space> address_space_const_ptr;

class GMAC_LOCAL address_space :
    public core::address_space,
    private gmac::util::mutex<address_space> {
    friend class resource_manager;
    friend class vdevice;

    typedef gmac::util::mutex<address_space> lock;

    hal::context_t &ctx_;

    hal::stream_t &streamLaunch_;
    hal::stream_t &streamToAccelerator_;
    hal::stream_t &streamToHost_;
    hal::stream_t &streamAccelerator_;

    process &proc_;

    typedef util::stl::locked_map<hostptr_t, hal::ptr_t> map_addresses;
    typedef util::stl::locked_map<hostptr_t, hal::buffer_t *> map_buffers;

    map_addresses mapDeviceAddresses_;
    map_buffers mapPinnedBuffers_;

    typedef std::map<gmac_kernel_id_t, kernel *> map_kernel;
    map_kernel kernels_;

    bool changes_;

public:
    address_space(hal::context_t &context,
                  hal::stream_t &streamLaunch,
                  hal::stream_t &streamToAccelerator,
                  hal::stream_t &streamToHost,
                  hal::stream_t &streamAccelerator,
                  process &proc);
    virtual ~address_space();

    gmacError_t map(hal::ptr_t &dst, hostptr_t src, size_t count, unsigned align = 1);
    gmacError_t unmap(hostptr_t addr, size_t count);

    hal::ptr_t alloc_host_pinned(size_t count, gmacError_t &err);
    gmacError_t free_host_pinned(hal::ptr_t ptr);

    hal::ptr_t get_host_pinned_mapping(hostptr_t ptr, gmacError_t &err);

    gmacError_t copy(hal::ptr_t dst, const hal::ptr_t src, size_t count);

    gmacError_t copy(hal::ptr_t dst, hal::device_input &input, size_t count);
    gmacError_t copy(hal::device_output &output, const hal::ptr_t src, size_t count);

    hal::event_ptr copy_async(hal::ptr_t dst, const hal::ptr_t src, size_t count, gmacError_t &err);

    hal::event_ptr copy_async(hal::ptr_t dst, hal::device_input &input, size_t count, gmacError_t &err);
    hal::event_ptr copy_async(hal::device_output &output, const hal::ptr_t src, size_t count, gmacError_t &err);

    gmacError_t memset(hal::ptr_t addr, int c, size_t count);
    hal::event_ptr memset_async(hal::ptr_t addr, int c, size_t count, gmacError_t &err);

    hal::context_t &get_hal_context();
    const hal::context_t &get_hal_context() const;

    bool is_integrated() const;

    bool has_direct_copy(core::address_space_const_ptr aspace) const;

    /**
     * Returns a kernel identified by k
     *
     * \param k A key that identifies the kernel object
     */
    kernel *get_kernel(gmac_kernel_id_t k);

#ifdef USE_OPENCL
#if 0
    gmacError_t acquire(hostptr_t addr);
    gmacError_t release(hostptr_t addr);
#endif
#endif

#if 0
    void notify_pending_changes();

    context &get_context();
#endif
};

}}}

#include "address_space-impl.h"

#endif
