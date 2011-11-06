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
WITH THE SOFTWARE.  */

#ifndef GMAC_CORE_HPE_VDEVICE_H_
#define GMAC_CORE_HPE_VDEVICE_H_

#include "config/common.h"

#include "core/vdevice.h"

#include "hal/types.h"

#ifdef USE_VM
#include "memory/vm/Bitmap.h"
#endif

#include "hal/types.h"

#include "util/NonCopyable.h"
#include "util/Reference.h"
#include "util/Private.h"

#include "kernel.h"

namespace __impl {

namespace memory {
    class object;
    class map_object;
    class Block;
}

namespace core {

namespace hpe {

class process;

class address_space;

/**
 * A vdevice represents the address space of a thread in an accelerator. Each
 * thread has one mode per accelerator type in the system
 */
class GMAC_LOCAL vdevice :
    public core::vdevice {
#if 0
    DBC_FORCE_TEST(vdevice)
#endif
    friend class ContextMap;
    //friend class Accelerator;
protected:
    process &proc_;

    util::smart_ptr<address_space>::shared aspace_;

    hal::stream_t &streamLaunch_;
#ifdef USE_VM
    __impl::memory::vm::Bitmap bitmap_;
#endif

#if 0
    virtual void reload() = 0;
    virtual Context &getContext() = 0;
    virtual void destroyContext(Context &context) const = 0;
#endif
    /**
     * Releases the resources used by the mode
     *
     * \return gmacSuccess on success. An error code on failure
    */
    TESTABLE gmacError_t cleanUp();
 
    /**
     * vdevice destructor
     */
    virtual ~vdevice();

public:
    /**
     * vdevice constructor
     *
     * \param proc Reference to the process which the mode belongs to
     * \param aspace Address space to which the virtual device is linked
     * \param streamLaunch Stream to which enqueue kernel launch:w
    */
    vdevice(process &proc,
            util::smart_ptr<address_space>::shared aspace,
            hal::stream_t &streamLaunch);

#if 0
    /**
     * Maps the given host memory on the accelerator memory
     * \param dst Reference to a pointer where to store the accelerator
     * address of the mapping
     * \param src Host address to be mapped
     * \param count Size of the mapping
     * \param align Alignment of the memory mapping. This value must be a
     * power of two
     * \return Error code
     */
    TESTABLE gmacError_t map(accptr_t &dst, hostptr_t src, size_t count, unsigned align = 1);

    /**
     * Unmaps the memory previously mapped by map
     * \param addr Host memory allocation to be unmap
     * \param count Size of the unmapping
     * \return Error code
     */
    TESTABLE gmacError_t unmap(hostptr_t addr, size_t count);

    /**
     * Copies data from system memory to accelerator memory
     * \param acc Destination accelerator pointer
     * \param host Source host pointer
     * \param count Number of bytes to be copied
     * \return Error code
     */
    TESTABLE gmacError_t copyToAccelerator(accptr_t acc, const hostptr_t host, size_t count);

    /**
     * Copies data from accelerator memory to system memory
     * \param host Destination host pointer
     * \param acc Source accelerator pointer
     * \param count Number of bytes to be copied
     * \return Error code
     */
    TESTABLE gmacError_t copyToHost(hostptr_t host, const accptr_t acc, size_t count);

    /** Copies data from accelerator memory to accelerator memory
     * \param dst Destination accelerator memory
     * \param src Source accelerator memory
     * \param count Number of bytes to be copied
     * \return Error code
     */
    TESTABLE gmacError_t copyAccelerator(accptr_t dst, const accptr_t src, size_t count);

    /**
     * Sets the contents of accelerator memory
     * \param addr Pointer to the accelerator memory to be set
     * \param c Value used to fill the memory
     * \param count Number of bytes to be set
     * \return Error code
     */
    TESTABLE gmacError_t memset(accptr_t addr, int c, size_t count);
#endif

    /**
     * Creates a KernelLaunch object that can be executed by the mode
     * \param k Handler of the kernel to be launched
     * \param config Configuration of the kernel execution
     * \param err gmacSucces on succes, an error code otherwise
     * \return Reference to the KernelLaunch object
     */
    kernel::launch *launch(gmac_kernel_id_t k, hal::kernel_t::config &conf, gmacError_t &err);

    /**
     * Executes a kernel using a KernelLaunch object
     * \param launch Reference to a KernelLaunch object
     * \return An event that represents the kernel execution
     */
    hal::event_t execute(kernel::launch &launch, gmacError_t &err);

    /**
     * Waits for kernel execution
     * \param launch Reference to KernelLaunch object
     * \return Error code
     */
    gmacError_t wait(kernel::launch &launch);

    /**
     * Waits for all kernels to finish execution
     * \return Error code
     */
    gmacError_t wait();

    /**
     * Returns the process which the mode belongs to
     * \return A reference to the process which the mode belongs to
     */
    process &get_process();

    /** Returns the process which the mode belongs to
     * \return A constant reference to the process which the mode belongs to
     */
    const process &get_process() const;

    /** Returns the memory information of the accelerator on which the mode runs
     * \param free A reference to a variable to store the memory available on the
     * accelerator
     * \param total A reference to a variable to store the total amount of memory
     * on the accelerator
     */
    void getMemInfo(size_t &free, size_t &total);

#ifdef USE_VM
    memory::vm::Bitmap &getDirtyBitmap();
    const memory::vm::Bitmap &getDirtyBitmap() const;
#endif

    /**
     * Waits for pending transfers before performing a kernel call
     *
     * \return gmacSuccess on success, an error code otherwise
     */
    //gmacError_t prepareForCall();

    hal::stream_t &eventStream();

    /**
     * Tells if the accelerator on which the vdevice is running shares memory with the CPU
     *
     * \return A boolean that tells if the accelerator on which the vdevice is running shares memory with the CPU
     */

    memory::map_object &get_object_map();
    const memory::map_object &get_object_map() const;

    util::smart_ptr<address_space>::shared get_address_space();
    util::smart_ptr<const address_space>::shared get_address_space() const;

    const hal::device &get_device();
};

}}}

#include "vdevice-impl.h"

#ifdef USE_DBC
#include "core/hpe/dbc/vdevice.h"
#endif

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
