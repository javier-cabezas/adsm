#ifndef GMAC_HAL_OPENCL_CONTEXT_H_
#define GMAC_HAL_OPENCL_CONTEXT_H_

#include <CL/cl.h>

#include <queue>

#include "hal/types-detail.h"

#include "util/unique.h"
#include "util/lock.h"

namespace __impl { namespace hal {
    
namespace opencl {

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

typedef hal::detail::list_event<device, backend_traits, implementation_traits> list_event_detail;

class GMAC_LOCAL list_event :
    public list_event_detail {
    typedef list_event_detail Parent;

public:
    gmacError_t sync();

    void set_synced();

    cl_event *get_event_array();

    size_t size() const;
};

class GMAC_LOCAL queue_event :
    std::queue<_event_t *>,
    gmac::util::mutex {

    typedef std::queue<_event_t *> Parent;

public:
    queue_event() :
        gmac::util::mutex("list_event")
    {
    }

    _event_t *pop()
    {
        _event_t *ret = NULL;

        lock();
        if (size() > 0) {
            ret = Parent::front();
            Parent::pop();
        }
        unlock();

        return ret;
    }

    void push(_event_t &event)
    {
        lock();
        Parent::push(&event);
        unlock();
    }
};

class GMAC_LOCAL context_t :
    public hal::detail::context_t<device, backend_traits, implementation_traits>,
    util::unique<context_t, GmacAddressSpaceId> {
    typedef hal::detail::context_t<device, backend_traits, implementation_traits> Parent;

    friend class stream_t;
    friend class _event_common_t;
    friend class event_t;
    friend class event_deleter;

    queue_event queueEvents_;

    _event_t *get_new_event(bool async, _event_t::type t);
    void dispose_event(_event_t &event);

public:
    context_t(cl_context ctx, device &device);

    accptr_t alloc(size_t count, gmacError_t &err);
    //hostptr_t alloc_host_pinned(hostptr_t &ptr, size_t count, GmacProtection hint, gmacError_t &err);
    buffer_t *alloc_buffer(size_t count, GmacProtection hint, gmacError_t &err);
    gmacError_t free(accptr_t acc);
    //gmacError_t free_host_pinned(hostptr_t ptr);
    gmacError_t free_buffer(buffer_t &buffer);

    event_t copy(accptr_t dst, hostptr_t src, size_t count, stream_t &stream, list_event_detail &dependencies, gmacError_t &err);
    event_t copy(accptr_t dst, hostptr_t src, size_t count, stream_t &stream, event_t event, gmacError_t &err);
    event_t copy(accptr_t dst, hostptr_t src, size_t count, stream_t &stream, gmacError_t &err);

    event_t copy(hostptr_t dst, accptr_t src, size_t count, stream_t &stream, list_event_detail &dependencies, gmacError_t &err);
    event_t copy(hostptr_t dst, accptr_t src, size_t count, stream_t &stream, event_t event, gmacError_t &err);
    event_t copy(hostptr_t dst, accptr_t src, size_t count, stream_t &stream, gmacError_t &err);

    event_t copy(accptr_t dst, accptr_t src, size_t count, stream_t &stream, list_event_detail &dependencies, gmacError_t &err);
    event_t copy(accptr_t dst, accptr_t src, size_t count, stream_t &stream, event_t event, gmacError_t &err);
    event_t copy(accptr_t dst, accptr_t src, size_t count, stream_t &stream, gmacError_t &err);

    event_t copy_async(accptr_t dst, buffer_t src, size_t off, size_t count, stream_t &stream, list_event_detail &dependencies, gmacError_t &err);
    event_t copy_async(accptr_t dst, buffer_t src, size_t off, size_t count, stream_t &stream, event_t event, gmacError_t &err);
    event_t copy_async(accptr_t dst, buffer_t src, size_t off, size_t count, stream_t &stream, gmacError_t &err);

    event_t copy_async(buffer_t dst, size_t off, accptr_t src, size_t count, stream_t &stream, list_event_detail &dependencies, gmacError_t &err);
    event_t copy_async(buffer_t dst, size_t off, accptr_t src, size_t count, stream_t &stream, event_t event, gmacError_t &err);
    event_t copy_async(buffer_t dst, size_t off, accptr_t src, size_t count, stream_t &stream, gmacError_t &err);

    event_t copy_async(accptr_t dst, accptr_t src, size_t count, stream_t &stream, list_event_detail &dependencies, gmacError_t &err);
    event_t copy_async(accptr_t dst, accptr_t src, size_t count, stream_t &stream, event_t event, gmacError_t &err);
    event_t copy_async(accptr_t dst, accptr_t src, size_t count, stream_t &stream, gmacError_t &err);

    event_t memset(accptr_t dst, int c, size_t count, stream_t &stream, list_event_detail &dependencies, gmacError_t &err);
    event_t memset(accptr_t dst, int c, size_t count, stream_t &stream, event_t event, gmacError_t &err);
    event_t memset(accptr_t dst, int c, size_t count, stream_t &stream, gmacError_t &err);

    event_t memset_async(accptr_t dst, int c, size_t count, stream_t &stream, list_event_detail &dependencies, gmacError_t &err);
    event_t memset_async(accptr_t dst, int c, size_t count, stream_t &stream, event_t event, gmacError_t &err);
    event_t memset_async(accptr_t dst, int c, size_t count, stream_t &stream, gmacError_t &err);

    accptr_t get_device_addr_from_pinned(hostptr_t addr);

    const code_repository &get_code_repository();
};

}}}

#endif /* GMAC_HAL_OPENCL_CONTEXT_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
