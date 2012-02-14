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

class aspace;

class code_repository;

typedef hal::detail::buffer hal_buffer;
typedef hal::detail::code_repository hal_code_repository;
typedef hal::detail::_event hal_event;
typedef hal::detail::event_ptr hal_event_ptr;
typedef hal::detail::stream hal_stream;

class GMAC_LOCAL aspace :
    public hal::detail::aspace,
    util::unique<aspace, GmacAddressSpaceId> {

    typedef hal::detail::aspace parent;

    friend class buffer_t;
    friend class _event_common_t;
    friend class event_deleter;
    //friend class detail::stream<implementation_traits>;
    friend class stream;

    CUcontext context_;

    hal_buffer *alloc_buffer(size_t size, GmacProtection hint, hal_stream &stream, gmacError_t &err);
    gmacError_t free_buffer(hal_buffer &buffer);
    
public:
    hal_event_ptr copy(hal::ptr dst, hal::const_ptr src, size_t count, hal_stream &s, list_event_detail *dependencies, gmacError_t &err);
    hal_event_ptr copy_async(hal::ptr dst, hal::const_ptr src, size_t count, hal_stream &s, list_event_detail *dependencies, gmacError_t &err);

    hal_event_ptr copy(hal::ptr dst, device_input &input, size_t count, hal_stream &s, list_event_detail *dependencies, gmacError_t &err);
    hal_event_ptr copy(device_output &output, hal::const_ptr src, size_t count, hal_stream &s, list_event_detail *dependencies, gmacError_t &err);
    hal_event_ptr memset(hal::ptr dst, int c, size_t count, hal_stream &s, list_event_detail *dependencies, gmacError_t &err);
    hal_event_ptr copy_async(hal::ptr dst, device_input &input, size_t count, hal_stream &s, list_event_detail *dependencies, gmacError_t &err);
    hal_event_ptr copy_async(device_output &output, hal::const_ptr src, size_t count, hal_stream &s, list_event_detail *dependencies, gmacError_t &err);
    hal_event_ptr memset_async(hal::ptr dst, int c, size_t count, hal_stream &s, list_event_detail *dependencies, gmacError_t &err);

    aspace(CUcontext ctx, device &device);

    hal::ptr alloc(size_t size, gmacError_t &err);
    hal::ptr alloc_host_pinned(size_t size, GmacProtection hint, gmacError_t &err);
    gmacError_t free(hal::ptr acc);
    gmacError_t free_host_pinned(hal::ptr ptr);

    hal::ptr get_device_addr_from_pinned(host_ptr addr);

    hal_code_repository &get_code_repository();

    void set();

    CUcontext &operator()();
    const CUcontext &operator()() const;

    _event_t *get_new_event(bool async, _event_t::type t);
    void dispose_event(_event_t &event);

};

class GMAC_LOCAL buffer_t :
    public hal::detail::buffer {
    typedef hal::detail::buffer parent;

    host_ptr addr_;

    aspace &get_aspace();
    const aspace &get_aspace() const;

public:
    buffer_t(host_ptr addr, size_t size, aspace &as);

    host_ptr get_addr();
    hal::ptr get_device_addr();
};

}}}

#endif /* GMAC_HAL_CUDA_CONTEXT_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
