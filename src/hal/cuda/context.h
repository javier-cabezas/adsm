#ifndef GMAC_HAL_CUDA_CONTEXT_H_
#define GMAC_HAL_CUDA_CONTEXT_H_

#include <cuda.h>
#include <driver_types.h>
#include <vector_types.h>

#include <queue>

#include "hal/types-detail.h"

#include "util/unique.h"
#include "util/lock.h"

namespace __impl { namespace hal {
    
namespace cuda {

class context_t;

class GMAC_LOCAL buffer_t :
    public hal::detail::buffer_t<implementation_traits> {
    typedef hal::detail::buffer_t<implementation_traits> Parent;

    hostptr_t addr_;

public:
    buffer_t(hostptr_t addr, size_t size, context_t &context);

    hostptr_t get_addr();
    ptr_t get_device_addr();
};

class code_repository;

class GMAC_LOCAL context_t :
    public hal::detail::context_t<device, backend_traits, implementation_traits>,
    util::unique<context_t, GmacAddressSpaceId> {
    typedef hal::detail::context_t<device, backend_traits, implementation_traits> Parent;

    friend class buffer_t;
    friend class _event_common_t;
    friend class event_ptr;
    friend class event_deleter;
    friend class detail::stream_t<backend_traits, implementation_traits>;
    friend class stream_t;

    _event_t *get_new_event(bool async, _event_t::type t);
    void dispose_event(_event_t &event);

    buffer_t *alloc_buffer(size_t size, GmacProtection hint, stream_t &stream, gmacError_t &err);
    gmacError_t free_buffer(buffer_t &buffer);

    event_ptr copy_backend(ptr_t dst, const ptr_t src, size_t count, stream_t &stream, list_event_detail *dependencies, gmacError_t &err);
    event_ptr copy_backend(ptr_t dst, device_input &input, size_t count, stream_t &stream, list_event_detail *dependencies, gmacError_t &err);
    event_ptr copy_backend(device_output &output, const ptr_t src, size_t count, stream_t &stream, list_event_detail *dependencies, gmacError_t &err);
    event_ptr memset_backend(ptr_t dst, int c, size_t count, stream_t &stream, list_event_detail *dependencies, gmacError_t &err);

    event_ptr copy_async_backend(ptr_t dst, const ptr_t src, size_t count, stream_t &stream, list_event_detail *dependencies, gmacError_t &err);
    event_ptr copy_async_backend(ptr_t dst, device_input &input, size_t count, stream_t &stream, list_event_detail *dependencies, gmacError_t &err);
    event_ptr copy_async_backend(device_output &output, const ptr_t src, size_t count, stream_t &stream, list_event_detail *dependencies, gmacError_t &err);
    event_ptr memset_async_backend(ptr_t dst, int c, size_t count, stream_t &stream, list_event_detail *dependencies, gmacError_t &err);
public:
    context_t(CUcontext ctx, device &device);

    ptr_t alloc(size_t size, gmacError_t &err);
    ptr_t alloc_host_pinned(size_t size, GmacProtection hint, gmacError_t &err);
    gmacError_t free(ptr_t acc);
    gmacError_t free_host_pinned(ptr_t ptr);

    ptr_t get_device_addr_from_pinned(hostptr_t addr);

    code_repository &get_code_repository();

    void set();
};

}}}

#endif /* GMAC_HAL_CUDA_CONTEXT_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
