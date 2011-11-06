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
    public hal::detail::buffer_t<device, backend_traits, implementation_traits> {
    typedef hal::detail::buffer_t<device, backend_traits, implementation_traits> Parent;

    hostptr_t addr_;

public:
    buffer_t(type t, hostptr_t addr, size_t size, context_t &context);

    void update(event_t &event);

    hostptr_t get_addr();
    accptr_t get_device_addr();
};

class code_repository;

typedef hal::detail::list_event<device, backend_traits, implementation_traits> list_event_detail;

class GMAC_LOCAL list_event :
    public list_event_detail {
    typedef list_event_detail Parent;

public:
    gmacError_t sync();
};

class GMAC_LOCAL queue_event :
    std::queue<_event_t *>,
    gmac::util::mutex {

    typedef std::queue<_event_t *> Parent;

public:
    queue_event();
    _event_t *pop();
    void push(_event_t &event);
};

template <typename T>
class GMAC_LOCAL map_pool :
    std::map<size_t, std::queue<T *> >,
    gmac::util::mutex {

    typedef std::queue<T *> queue_context;
    typedef std::map<size_t, queue_context> Parent;

public:
    map_pool() :
        gmac::util::mutex("map_pool")
    {}

    T *pop(size_t size);
    void push(T *event);
};

typedef map_pool<buffer_t> map_buffer;
typedef map_pool<void> map_memory;

class GMAC_LOCAL context_t :
    public hal::detail::context_t<device, backend_traits, implementation_traits>,
    util::unique<context_t, GmacAddressSpaceId> {
    typedef hal::detail::context_t<device, backend_traits, implementation_traits> Parent;

    friend class buffer_t;
    friend class _event_common_t;
    friend class event_t;
    friend class event_deleter;
    friend class stream_t;

    queue_event queueEvents_;

    map_buffer mapBuffersIn_;
    map_buffer mapBuffersOut_;
    
    map_memory mapMemory_;

    hostptr_t get_memory(size_t size);
    void put_memory(void *ptr, size_t size);
    buffer_t &get_input_buffer(size_t size);
    buffer_t &get_output_buffer(size_t size);
    void put_input_buffer(buffer_t &buffer);
    void put_output_buffer(buffer_t &buffer);

    _event_t *get_new_event(bool async, _event_t::type t);
    void dispose_event(_event_t &event);

    buffer_t *alloc_buffer(size_t size, GmacProtection hint, gmacError_t &err);
    gmacError_t free_buffer(buffer_t &buffer);
public:
    context_t(CUcontext ctx, device &device);

    accptr_t alloc(size_t size, gmacError_t &err);
    hostptr_t alloc_host_pinned(size_t size, GmacProtection hint, gmacError_t &err);
    gmacError_t free(accptr_t acc);
    gmacError_t free_host_pinned(hostptr_t ptr);

    event_t copy(accptr_t dst, hostptr_t src, size_t count, stream_t &stream, list_event_detail &dependencies, gmacError_t &err);
    event_t copy(accptr_t dst, hostptr_t src, size_t count, stream_t &stream, event_t event, gmacError_t &err);
    event_t copy(accptr_t dst, hostptr_t src, size_t count, stream_t &stream, gmacError_t &err);

    event_t copy(hostptr_t dst, accptr_t src, size_t count, stream_t &stream, list_event_detail &dependencies, gmacError_t &err);
    event_t copy(hostptr_t dst, accptr_t src, size_t count, stream_t &stream, event_t event, gmacError_t &err);
    event_t copy(hostptr_t dst, accptr_t src, size_t count, stream_t &stream, gmacError_t &err);

    event_t copy(accptr_t dst, accptr_t src, size_t count, stream_t &stream, list_event_detail &dependencies, gmacError_t &err);
    event_t copy(accptr_t dst, accptr_t src, size_t count, stream_t &stream, event_t event, gmacError_t &err);
    event_t copy(accptr_t dst, accptr_t src, size_t count, stream_t &stream, gmacError_t &err);

    event_t copy(accptr_t dst, device_input &input, size_t count, stream_t &stream, list_event_detail &dependencies, gmacError_t &err);
    event_t copy(accptr_t dst, device_input &input, size_t count, stream_t &stream, event_t event, gmacError_t &err);
    event_t copy(accptr_t dst, device_input &input, size_t count, stream_t &stream, gmacError_t &err);

    event_t copy(device_output &output, accptr_t src, size_t count, stream_t &stream, list_event_detail &dependencies, gmacError_t &err);
    event_t copy(device_output &output, accptr_t src, size_t count, stream_t &stream, event_t event, gmacError_t &err);
    event_t copy(device_output &output, accptr_t src, size_t count, stream_t &stream, gmacError_t &err);
 
    event_t copy_async(accptr_t dst, buffer_t src, size_t off, size_t count, stream_t &stream, list_event_detail &dependencies, gmacError_t &err);
    event_t copy_async(accptr_t dst, buffer_t src, size_t off, size_t count, stream_t &stream, event_t event, gmacError_t &err);
    event_t copy_async(accptr_t dst, buffer_t src, size_t off, size_t count, stream_t &stream, gmacError_t &err);

    event_t copy_async(buffer_t dst, size_t off, accptr_t src, size_t count, stream_t &stream, list_event_detail &dependencies, gmacError_t &err);
    event_t copy_async(buffer_t dst, size_t off, accptr_t src, size_t count, stream_t &stream, event_t event, gmacError_t &err);
    event_t copy_async(buffer_t dst, size_t off, accptr_t src, size_t count, stream_t &stream, gmacError_t &err);

    event_t copy_async(accptr_t dst, accptr_t src, size_t count, stream_t &stream, list_event_detail &dependencies, gmacError_t &err);
    event_t copy_async(accptr_t dst, accptr_t src, size_t count, stream_t &stream, event_t event, gmacError_t &err);
    event_t copy_async(accptr_t dst, accptr_t src, size_t count, stream_t &stream, gmacError_t &err);

    event_t copy_async(accptr_t dst, device_input &input, size_t count, stream_t &stream, list_event_detail &dependencies, gmacError_t &err);
    event_t copy_async(accptr_t dst, device_input &input, size_t count, stream_t &stream, event_t event, gmacError_t &err);
    event_t copy_async(accptr_t dst, device_input &input, size_t count, stream_t &stream, gmacError_t &err);

    event_t copy_async(device_output &output, accptr_t src, size_t count, stream_t &stream, list_event_detail &dependencies, gmacError_t &err);
    event_t copy_async(device_output &output, accptr_t src, size_t count, stream_t &stream, event_t event, gmacError_t &err);
    event_t copy_async(device_output &output, accptr_t src, size_t count, stream_t &stream, gmacError_t &err);
 
    event_t memset(accptr_t dst, int c, size_t count, stream_t &stream, list_event_detail &dependencies, gmacError_t &err);
    event_t memset(accptr_t dst, int c, size_t count, stream_t &stream, event_t event, gmacError_t &err);
    event_t memset(accptr_t dst, int c, size_t count, stream_t &stream, gmacError_t &err);

    event_t memset_async(accptr_t dst, int c, size_t count, stream_t &stream, list_event_detail &dependencies, gmacError_t &err);
    event_t memset_async(accptr_t dst, int c, size_t count, stream_t &stream, event_t event, gmacError_t &err);
    event_t memset_async(accptr_t dst, int c, size_t count, stream_t &stream, gmacError_t &err);

    accptr_t get_device_addr_from_pinned(hostptr_t addr);

    const code_repository &get_code_repository();

    void set();
};

}}}

#endif /* GMAC_HAL_CUDA_CONTEXT_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
