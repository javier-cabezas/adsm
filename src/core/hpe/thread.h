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

#ifndef GMAC_CORE_HPE_THREAD_H_
#define GMAC_CORE_HPE_THREAD_H_

#include "core/thread.h"
#include "util/Private.h"

#include "hal/types.h"

namespace __impl { namespace core { namespace hpe {

class address_space;
class context;
class process;
class vdevice;

#if 0
class GMAC_LOCAL contextMap : protected std::map<address_space *, context *> {
protected:
    typedef std::map<THREAD_T, context *> Parent;
    address_space &owner_;

public:
    contextMap(address_space &owner);
    void add(THREAD_T id, context &ctx);
    context *find(THREAD_T id);
    void remove(THREAD_T id);
    void clean();
};
#endif

/** Contains some thread-dependent values */
class GMAC_LOCAL thread :
    public core::thread {
    friend class process;
    friend class resource_manager;

private:
    process &process_;

    typedef std::map<GmacVirtualDeviceId, vdevice *> map_vdevice;
    typedef std::map<address_space *, context *> map_context;
    typedef std::map<vdevice *, hal::kernel_t::config *> map_config;

    vdevice *currentVirtualDevice_;

    map_vdevice mapVDevices_;
    map_context mapContexts_;
    map_config mapConfigs_;

    static bool has_current_thread();
    //static vdevice_table &getCurrentVirtualDeviceTable();

public:
    thread(process &proc);
    virtual ~thread();

    // Virtual devices
    vdevice *get_virtual_device(GmacVirtualDeviceId id);
    vdevice &get_current_virtual_device() const;
    gmacError_t add_virtual_device(GmacVirtualDeviceId id, vdevice &dev);
    gmacError_t remove_virtual_device(vdevice &dev);
    void set_current_virtual_device(vdevice &dev);

    // contexts
    context *get_context(address_space &aspace);
    gmacError_t set_context(address_space &aspace, context *context);

    // Kernel configuration
    gmacError_t new_kernel_config(hal::kernel_t::config &config);
    gmacError_t new_kernel_config(hal::kernel_t::config &config, vdevice &dev);
    hal::kernel_t::config &get_kernel_config();
    hal::kernel_t::config &get_kernel_config(vdevice &dev);

    static thread &get_current_thread();
};

}}}

#include "thread-impl.h"

#ifdef USE_DBC
namespace __dbc { namespace core { namespace hpe {
    typedef __impl::core::hpe::thread thread;
}}}
#endif

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
