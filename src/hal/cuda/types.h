#ifndef GMAC_HAL_CUDA_TYPES_H_
#define GMAC_HAL_CUDA_TYPES_H_

#include <cuda.h>
#include <driver_types.h>
#include <vector_types.h>

#include "hal/types-detail.h"

#include "util/Unique.h"

namespace __impl { namespace hal {
    
namespace cuda {

class device;

class coherence_domain;
class context_t;
class stream_t;
class kernel_t;
class texture_t;
class variable_t;
class code_repository;
class event_t;
class async_event_t;
class buffer_t;

typedef hal::detail::backend_traits<CUcontext, CUstream, CUevent, CUfunction, dim3> backend_traits;
typedef hal::detail::implementation_traits<coherence_domain, context_t, stream_t, kernel_t, texture_t, variable_t, code_repository, event_t, async_event_t, buffer_t> implementation_traits;

gmacError_t error(CUresult err);

class context_t;

class GMAC_LOCAL buffer_t :
    public hal::detail::buffer_t<device, backend_traits, implementation_traits> {
    typedef hal::detail::buffer_t<device, backend_traits, implementation_traits> Parent;

    hostptr_t addr_;

public:
    buffer_t(hostptr_t addr, context_t &context);

    hostptr_t get_addr();
    accptr_t get_device_addr();
};

class code_repository;

class GMAC_LOCAL context_t :
    public hal::detail::context_t<device, backend_traits, implementation_traits>,
    util::unique<context_t, GmacAddressSpaceId> {
    typedef hal::detail::context_t<device, backend_traits, implementation_traits> Parent;
    typedef hal::detail::list_event<device, backend_traits, implementation_traits> list_event;

    friend class stream_t;
    friend class _event_common_t;
    friend class event_t;
    friend class async_event_t;

public:
    context_t(CUcontext ctx, device &device);

    accptr_t alloc(size_t count, gmacError_t &err);
    //hostptr_t alloc_host_pinned(hostptr_t &ptr, size_t count, GmacProtection hint, gmacError_t &err);
    buffer_t *alloc_buffer(size_t count, GmacProtection hint, gmacError_t &err);
    gmacError_t free(accptr_t acc);
    //gmacError_t free_host_pinned(hostptr_t ptr);
    gmacError_t free_buffer(buffer_t &buffer);

    event_t &copy(accptr_t dst, hostptr_t src, size_t count, stream_t &stream, list_event &dependencies = list_event::empty);
    event_t &copy(accptr_t dst, hostptr_t src, size_t count, stream_t &stream, async_event_t &event);
    event_t &copy(hostptr_t dst, accptr_t src, size_t count, stream_t &stream, list_event &dependencies = list_event::empty);
    event_t &copy(hostptr_t dst, accptr_t src, size_t count, stream_t &stream, async_event_t &event);
    event_t &copy(accptr_t dst, accptr_t src, size_t count, stream_t &stream, list_event &dependencies = list_event::empty);
    event_t &copy(accptr_t dst, accptr_t src, size_t count, stream_t &stream, async_event_t &event);

    async_event_t &copy_async(accptr_t dst, buffer_t src, size_t off, size_t count, stream_t &stream, list_event &dependencies = list_event::empty);
    async_event_t &copy_async(accptr_t dst, buffer_t src, size_t off, size_t count, stream_t &stream, async_event_t &event);
    async_event_t &copy_async(buffer_t dst, size_t off, accptr_t src, size_t count, stream_t &stream, list_event &dependencies = list_event::empty);
    async_event_t &copy_async(buffer_t dst, size_t off, accptr_t src, size_t count, stream_t &stream, async_event_t &event);
    async_event_t &copy_async(accptr_t dst, accptr_t src, size_t count, stream_t &stream, list_event &dependencies = list_event::empty);
    async_event_t &copy_async(accptr_t dst, accptr_t src, size_t count, stream_t &stream, async_event_t &event);

    event_t &memset(accptr_t dst, int c, size_t count, stream_t &stream, list_event &dependencies = list_event::empty);
    event_t &memset(accptr_t dst, int c, size_t count, stream_t &stream, async_event_t &event);
    async_event_t &memset_async(accptr_t dst, int c, size_t count, stream_t &stream, list_event &dependencies = list_event::empty);
    async_event_t &memset_async(accptr_t dst, int c, size_t count, stream_t &stream, async_event_t &event);

    accptr_t get_device_addr_from_pinned(hostptr_t addr);

    const code_repository &get_code_repository() const;

    void set();
};

class GMAC_LOCAL stream_t :
    public hal::detail::stream_t<device, backend_traits, implementation_traits> {
    typedef hal::detail::stream_t<device, backend_traits, implementation_traits> Parent;
public:
    stream_t(CUstream stream, context_t &context);

    context_t &get_context();

    Parent::state query();
    gmacError_t sync();
};

class GMAC_LOCAL kernel_t :
    public hal::detail::kernel_t<device, backend_traits, implementation_traits> {

    typedef hal::detail::kernel_t<device, backend_traits, implementation_traits> Parent;
    typedef hal::detail::list_event<device, backend_traits, implementation_traits> list_event;

public:
    class launch;

    class GMAC_LOCAL config :
        public hal::detail::kernel_t<device, backend_traits, implementation_traits>::config {
        friend class launch;

        unsigned nArgs_;

        dim3 dimsGlobal_;
        dim3 dimsGroup_;
        size_t memShared_;

        const void *params_[256];
    public:
        config(dim3 global, dim3 group, size_t shared, cudaStream_t tokens);

        unsigned get_nargs() const;
        const dim3 &get_dims_global() const;
        const dim3 &get_dims_group() const;

        gmacError_t set_arg(const void *arg, size_t size, unsigned index);
        gmacError_t register_kernel();
    };

    class GMAC_LOCAL launch :
        public hal::detail::kernel_t<device, backend_traits, implementation_traits>::launch {

    public:
        launch(kernel_t &parent, Parent::config &conf, stream_t &stream);

        async_event_t &execute(list_event &dependencies = list_event::empty);
        async_event_t &execute(async_event_t &event);
    };

    kernel_t(CUfunction func, const std::string &name);

    launch &launch_config(Parent::config &conf, stream_t &stream);
};

class GMAC_LOCAL _event_common_t {
    friend class device;
    friend class context_t;
    friend class kernel_t;
    stream_t *stream_;

protected:
    CUevent eventStart_;
    CUevent eventEnd_;

    hal::time_t timeBase_;

    // Not instantiable
    _event_common_t();

    void begin(stream_t &stream);
    void end();

    stream_t &get_stream();
};

class GMAC_LOCAL event_t :
    public hal::detail::event_t<device, backend_traits, implementation_traits>,
    public _event_common_t {
    friend class device;
    friend class context_t;
    typedef hal::detail::event_t<device, backend_traits, implementation_traits> Parent;

public:
    event_t(Parent::type t, context_t &context);
};

class GMAC_LOCAL async_event_t :
    public hal::detail::async_event_t<device, backend_traits, implementation_traits>,
    public _event_common_t {
    friend class device;

    typedef hal::detail::async_event_t<device, backend_traits, implementation_traits> Parent;

public:
    async_event_t(Parent::type t, context_t &context);
    gmacError_t sync();

    Parent::state get_state();
};


}}}

#include "context-impl.h"
#include "stream-impl.h"
#include "kernel-impl.h"
#include "event-impl.h"

#include "module.h"

#endif /* GMAC_HAL_CUDA_TYPES_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
