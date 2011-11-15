#ifndef GMAC_HAL_TYPES_ASPACE_H_
#define GMAC_HAL_TYPES_ASPACE_H_

#include <list>
#include <queue>

#include "util/lock.h"
#include "util/Logger.h"

namespace __impl { namespace hal {

namespace detail {

template <typename I>
class GMAC_LOCAL buffer_t {
public:
    enum type {
        ToHost,
        ToDevice,
        DeviceToDevice
    };
private:
    typename I::context &context_;
    size_t size_;

    typename I::event event_;

protected:
    buffer_t(size_t size, typename I::context &context);

    type get_type() const;

public:
    virtual hostptr_t get_addr() = 0;
    virtual typename I::ptr get_device_addr() = 0;
    typename I::context &get_context();
    const typename I::context &get_context() const;
    size_t get_size() const;

    gmacError_t wait()
    {
        gmacError_t ret = gmacSuccess;
        if (event_.is_valid()) {
            ret = event_.sync();
        }

        return ret;
    }
};

template <typename I>
class GMAC_LOCAL list_event :
    protected std::list<typename I::event> {

    typedef std::list<typename I::event> Parent;
public:
    static list_event empty;

    void add_event(typename I::event event);
};

template <typename I>
list_event<I> list_event<I>::empty;

template <typename D, typename B, typename I>
class GMAC_LOCAL code_repository
{
public:
    virtual const typename I::kernel *get_kernel(gmac_kernel_id_t key) const = 0;
    virtual const typename I::kernel *get_kernel(const std::string &name) const = 0;
};

template <typename I>
class GMAC_LOCAL queue_event :
    std::queue<typename I::event::event_type *>,
    gmac::util::mutex {

    typedef std::queue<typename I::event::event_type *> Parent;

public:
    queue_event();
    typename I::event::event_type *pop();
    void push(typename I::event::event_type &event);
};

template <typename D, typename B, typename I>
class GMAC_LOCAL context_t {
private:
    typedef map_pool<void> map_memory;
    typedef map_pool<typename I::buffer> map_buffer;

protected:
    typename B::context context_;
    D &device_;

    static const unsigned MaxBuffersIn_  = 3;
    static const unsigned MaxBuffersOut_ = 3;

    unsigned nBuffersIn_;
    unsigned nBuffersOut_;

    map_memory mapMemory_;

    map_buffer mapFreeBuffersIn_;
    map_buffer mapFreeBuffersOut_;

    map_buffer mapUsedBuffersIn_;
    map_buffer mapUsedBuffersOut_;

    queue_event<I> queueEvents_;

    hostptr_t get_memory(size_t size);
    void put_memory(void *ptr, size_t size);

    typename I::buffer &get_input_buffer(size_t size);
    void put_input_buffer(typename I::buffer &buffer);
 
    typename I::buffer &get_output_buffer(size_t size);
    void put_output_buffer(typename I::buffer &buffer);

    virtual typename I::buffer *alloc_buffer(size_t size, GmacProtection hint, gmacError_t &err) = 0;
    virtual gmacError_t free_buffer(typename I::buffer &buffer) = 0;

    context_t(typename B::context context, D &device);

    virtual typename I::event copy_backend(typename I::ptr dst, const typename I::ptr src, size_t count, typename I::stream &stream, list_event<I> *dependencies, gmacError_t &err) = 0;
    virtual typename I::event copy_backend(typename I::ptr dst, device_input &input, size_t count, typename I::stream &stream, list_event<I> *dependencies, gmacError_t &err) = 0;
    virtual typename I::event copy_backend(device_output &output, const typename I::ptr src, size_t count, typename I::stream &stream, list_event<I> *dependencies, gmacError_t &err) = 0;
    virtual typename I::event memset_backend(typename I::ptr dst, int c, size_t count, typename I::stream &stream, list_event<I> *dependencies, gmacError_t &err) = 0;

    virtual typename I::event copy_async_backend(typename I::ptr dst, const typename I::ptr src, size_t count, typename I::stream &stream, list_event<I> *dependencies, gmacError_t &err) = 0;
    virtual typename I::event copy_async_backend(typename I::ptr dst, device_input &input, size_t count, typename I::stream &stream, list_event<I> *dependencies, gmacError_t &err) = 0;
    virtual typename I::event copy_async_backend(device_output &output, const typename I::ptr src, size_t count, typename I::stream &stream, list_event<I> *dependencies, gmacError_t &err) = 0;
    virtual typename I::event memset_async_backend(typename I::ptr dst, int c, size_t count, typename I::stream &stream, list_event<I> *dependencies, gmacError_t &err) = 0;

public:
    D &get_device();
    const D &get_device() const;

    typename B::context &operator()();
    const typename B::context &operator()() const;

    virtual typename I::ptr alloc(size_t size, gmacError_t &err) = 0;
    virtual typename I::ptr alloc_host_pinned(size_t size, GmacProtection hint, gmacError_t &err) = 0;

    virtual gmacError_t free(typename I::ptr acc) = 0;
    virtual gmacError_t free_host_pinned(typename I::ptr ptr) = 0;

    typename I::event copy(typename I::ptr dst, const typename I::ptr src, size_t count, typename I::stream &stream, list_event<I> &dependencies, gmacError_t &err);
    typename I::event copy(typename I::ptr dst, const typename I::ptr src, size_t count, typename I::stream &stream, typename I::event event, gmacError_t &err);
    typename I::event copy(typename I::ptr dst, const typename I::ptr src, size_t count, typename I::stream &stream, gmacError_t &err);

    typename I::event copy(typename I::ptr dst, device_input &input, size_t count, typename I::stream &stream, list_event<I> &dependencies, gmacError_t &err);
    typename I::event copy(typename I::ptr dst, device_input &input, size_t count, typename I::stream &stream, typename I::event event, gmacError_t &err);
    typename I::event copy(typename I::ptr dst, device_input &input, size_t count, typename I::stream &stream, gmacError_t &err);

    typename I::event copy(device_output &output, const typename I::ptr src, size_t count, typename I::stream &stream, list_event<I> &dependencies, gmacError_t &err);
    typename I::event copy(device_output &output, const typename I::ptr src, size_t count, typename I::stream &stream, typename I::event event, gmacError_t &err);
    typename I::event copy(device_output &output, const typename I::ptr src, size_t count, typename I::stream &stream, gmacError_t &err);

    typename I::event copy_async(typename I::ptr dst, const typename I::ptr src, size_t count, typename I::stream &stream, list_event<I> &dependencies, gmacError_t &err);
    typename I::event copy_async(typename I::ptr dst, const typename I::ptr src, size_t count, typename I::stream &stream, typename I::event event, gmacError_t &err);
    typename I::event copy_async(typename I::ptr dst, const typename I::ptr src, size_t count, typename I::stream &stream, gmacError_t &err);

    typename I::event copy_async(typename I::ptr dst, device_input &input, size_t count, typename I::stream &stream, list_event<I> &dependencies, gmacError_t &err);
    typename I::event copy_async(typename I::ptr dst, device_input &input, size_t count, typename I::stream &stream, typename I::event event, gmacError_t &err);
    typename I::event copy_async(typename I::ptr dst, device_input &input, size_t count, typename I::stream &stream, gmacError_t &err);

    typename I::event copy_async(device_output &output, const typename I::ptr src, size_t count, typename I::stream &stream, list_event<I> &dependencies, gmacError_t &err);
    typename I::event copy_async(device_output &output, const typename I::ptr src, size_t count, typename I::stream &stream, typename I::event event, gmacError_t &err);
    typename I::event copy_async(device_output &output, const typename I::ptr src, size_t count, typename I::stream &stream, gmacError_t &err);

    typename I::event memset(typename I::ptr dst, int c, size_t count, typename I::stream &stream, list_event<I> &dependencies, gmacError_t &err);
    typename I::event memset(typename I::ptr dst, int c, size_t count, typename I::stream &stream, typename I::event event, gmacError_t &err);
    typename I::event memset(typename I::ptr dst, int c, size_t count, typename I::stream &stream, gmacError_t &err);

    typename I::event memset_async(typename I::ptr dst, int c, size_t count, typename I::stream &stream, list_event<I> &dependencies, gmacError_t &err);
    typename I::event memset_async(typename I::ptr dst, int c, size_t count, typename I::stream &stream, typename I::event event, gmacError_t &err);
    typename I::event memset_async(typename I::ptr dst, int c, size_t count, typename I::stream &stream, gmacError_t &err);

    virtual const typename I::code_repository &get_code_repository() = 0;
};

}

}}

#endif /* ASPACE_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
