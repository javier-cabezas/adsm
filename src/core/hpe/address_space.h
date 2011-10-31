/* Copyright (c) 2009, 2010, 2011 University of Illinois
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

#include "util/Lock.h"

namespace __impl {
    
namespace core {

class io_buffer;
    
namespace hpe {

class address_space;
class context;
class kernel;
class process;
class vdevice;

class GMAC_LOCAL address_space :
    public core::address_space {
    friend class resource_manager;
    friend class vdevice;

    hal::context_t &ctx_;

    hal::stream_t &streamLaunch_;
    hal::stream_t &streamToAccelerator_;
    hal::stream_t &streamToHost_;
    hal::stream_t &streamAccelerator_;

    process &proc_;

    typedef std::map<hostptr_t, accptr_t> DeviceAddresses;
    typedef std::map<hostptr_t, hal::buffer_t *> PinnedBuffers;

    DeviceAddresses deviceAddresses_;
    PinnedBuffers pinnedBuffers_;

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

    core::io_buffer *create_io_buffer(size_t count, GmacProtection prot);
    gmacError_t destroy_io_buffer(core::io_buffer &buffer);

    gmacError_t map(accptr_t &dst, hostptr_t src, size_t count, unsigned align = 1);
    gmacError_t unmap(hostptr_t addr, size_t count);

    hostptr_t alloc_host_pinned(size_t count, gmacError_t &err);
    gmacError_t free_host_pinned(hostptr_t ptr);

    accptr_t get_host_pinned_mapping(hostptr_t ptr, gmacError_t &err);

    gmacError_t copy(accptr_t acc, const hostptr_t host, size_t count);

    gmacError_t copy(hostptr_t host, const accptr_t acc, size_t count);

    gmacError_t copy(accptr_t dst, const accptr_t src, size_t count);

    gmacError_t copy(accptr_t dst, core::io_buffer &buffer, size_t off, size_t count);

    gmacError_t copy(core::io_buffer &buffer, size_t off, const accptr_t src, size_t count);

    gmacError_t memset(accptr_t addr, int c, size_t count);

    hal::context_t &get_hal_context();
    const hal::context_t &get_hal_context() const;

    bool is_integrated() const;

    bool has_direct_copy(const core::address_space &aspace) const;

    /**
     * Registers a new kernel that can be executed by the owner thread of the mode
     *
     * \param k A key that identifies the kernel object
     * \param ker A const reference to the kernel to be registered
     */
    void register_kernel(gmac_kernel_id_t k, const hal::kernel_t &ker);

    /**
     * Returns a kernel identified by k
     *
     * \param k A key that identifies the kernel object
     */
    kernel *get_kernel(gmac_kernel_id_t k);


#if 0
    void notify_pending_changes();
#endif

    context &get_context();
};

}}}

#include "address_space-impl.h"

#endif
