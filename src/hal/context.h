#ifndef GMAC_HAL_TYPES_ASPACE_H_
#define GMAC_HAL_TYPES_ASPACE_H_

#include <list>
#include <queue>

#include "trace/logger.h"

#include "util/attribute.h"
#include "util/lock.h"
#include "util/locked_counter.h"
#include "util/trigger.h"

namespace __impl { namespace hal {

namespace detail {

template <typename I>
class list_event;

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

    typename I::event_ptr event_;

protected:
    buffer_t(size_t size, typename I::context &context);

    type get_type() const;

public:
    virtual host_ptr get_addr() = 0;
    virtual typename I::ptr get_device_addr() = 0;
    typename I::context &get_context();
    const typename I::context &get_context() const;
    size_t get_size() const;

    void set_event(typename I::event_ptr event)
    {
        event_ = event;
    }

    gmacError_t wait()
    {
        gmacError_t ret = gmacSuccess;
        if (event_) {
            ret = event_->sync();
        }

        return ret;
    }
};

template <typename D, typename B, typename I>
class GMAC_LOCAL code_repository
{
public:
    virtual typename I::kernel *get_kernel(gmac_kernel_id_t key) = 0;
    virtual typename I::kernel *get_kernel(const std::string &name) = 0;
};

template <typename I>
class GMAC_LOCAL queue_event :
    std::queue<typename I::event *>,
    gmac::util::spinlock<queue_event<I> > {

    typedef std::queue<typename I::event *> Parent;
    typedef gmac::util::spinlock<queue_event<I> > Lock;

public:
    queue_event();
    typename I::event *pop();
    void push(typename I::event &event);
};

template <typename D, typename B, typename I>
class GMAC_LOCAL context_t :
    public util::attributes<context_t<D, B, I> >,
    public util::on_construction<typename I::context>,
    public util::on_destruction<typename I::context> {

private:
    typedef map_pool<void> map_memory;
    typedef map_pool<typename I::buffer> map_buffer;

protected:
    typename B::context context_;
    typedef util::locked_counter<unsigned, gmac::util::spinlock<context_t> > buffer_counter;
    D &device_;

    static const unsigned &MaxBuffersIn_;
    static const unsigned &MaxBuffersOut_;

    buffer_counter nBuffersIn_;
    buffer_counter nBuffersOut_;

    map_memory mapMemory_;

    map_buffer mapFreeBuffersIn_;
    map_buffer mapFreeBuffersOut_;

    map_buffer mapUsedBuffersIn_;
    map_buffer mapUsedBuffersOut_;

    queue_event<I> queueEvents_;

    host_ptr get_memory(size_t size);
    void put_memory(void *ptr, size_t size);

    typename I::buffer *get_input_buffer(size_t size, typename I::stream &stream, typename I::event_ptr event);
    void put_input_buffer(typename I::buffer &buffer);
 
    typename I::buffer *get_output_buffer(size_t size, typename I::stream &stream, typename I::event_ptr event);
    void put_output_buffer(typename I::buffer &buffer);

    virtual typename I::buffer *alloc_buffer(size_t size, GmacProtection hint, typename I::stream &stream, gmacError_t &err) = 0;
    virtual gmacError_t free_buffer(typename I::buffer &buffer) = 0;

    context_t(typename B::context context, D &device);

    virtual typename I::event_ptr copy_backend(typename I::ptr dst, typename I::ptr_const src, size_t count, typename I::stream &stream, list_event<I> *dependencies, gmacError_t &err) = 0;
    virtual typename I::event_ptr copy_backend(typename I::ptr dst, device_input &input, size_t count, typename I::stream &stream, list_event<I> *dependencies, gmacError_t &err) = 0;
    virtual typename I::event_ptr copy_backend(device_output &output, typename I::ptr_const src, size_t count, typename I::stream &stream, list_event<I> *dependencies, gmacError_t &err) = 0;
    virtual typename I::event_ptr memset_backend(typename I::ptr dst, int c, size_t count, typename I::stream &stream, list_event<I> *dependencies, gmacError_t &err) = 0;

    virtual typename I::event_ptr copy_async_backend(typename I::ptr dst, typename I::ptr_const src, size_t count, typename I::stream &stream, list_event<I> *dependencies, gmacError_t &err) = 0;
    virtual typename I::event_ptr copy_async_backend(typename I::ptr dst, device_input &input, size_t count, typename I::stream &stream, list_event<I> *dependencies, gmacError_t &err) = 0;
    virtual typename I::event_ptr copy_async_backend(device_output &output, typename I::ptr_const src, size_t count, typename I::stream &stream, list_event<I> *dependencies, gmacError_t &err) = 0;
    virtual typename I::event_ptr memset_async_backend(typename I::ptr dst, int c, size_t count, typename I::stream &stream, list_event<I> *dependencies, gmacError_t &err) = 0;

public:
    D &get_device();
    const D &get_device() const;

    typename B::context &operator()();
    const typename B::context &operator()() const;

    virtual typename I::ptr alloc(size_t size, gmacError_t &err) = 0;
    virtual typename I::ptr alloc_host_pinned(size_t size, GmacProtection hint, gmacError_t &err) = 0;

    virtual gmacError_t free(typename I::ptr acc) = 0;
    virtual gmacError_t free_host_pinned(typename I::ptr ptr) = 0;

    typename I::event_ptr copy(typename I::ptr dst, typename I::ptr_const src, size_t count, typename I::stream &stream, list_event<I> &dependencies, gmacError_t &err);
    typename I::event_ptr copy(typename I::ptr dst, typename I::ptr_const src, size_t count, typename I::stream &stream, typename I::event_ptr event, gmacError_t &err);
    typename I::event_ptr copy(typename I::ptr dst, typename I::ptr_const src, size_t count, typename I::stream &stream, gmacError_t &err);

    typename I::event_ptr copy(typename I::ptr dst, device_input &input, size_t count, typename I::stream &stream, list_event<I> &dependencies, gmacError_t &err);
    typename I::event_ptr copy(typename I::ptr dst, device_input &input, size_t count, typename I::stream &stream, typename I::event_ptr event, gmacError_t &err);
    typename I::event_ptr copy(typename I::ptr dst, device_input &input, size_t count, typename I::stream &stream, gmacError_t &err);

    typename I::event_ptr copy(device_output &output, typename I::ptr_const src, size_t count, typename I::stream &stream, list_event<I> &dependencies, gmacError_t &err);
    typename I::event_ptr copy(device_output &output, typename I::ptr_const src, size_t count, typename I::stream &stream, typename I::event_ptr event, gmacError_t &err);
    typename I::event_ptr copy(device_output &output, typename I::ptr_const src, size_t count, typename I::stream &stream, gmacError_t &err);

    typename I::event_ptr copy_async(typename I::ptr dst, typename I::ptr_const src, size_t count, typename I::stream &stream, list_event<I> &dependencies, gmacError_t &err);
    typename I::event_ptr copy_async(typename I::ptr dst, typename I::ptr_const src, size_t count, typename I::stream &stream, typename I::event_ptr event, gmacError_t &err);
    typename I::event_ptr copy_async(typename I::ptr dst, typename I::ptr_const src, size_t count, typename I::stream &stream, gmacError_t &err);

    typename I::event_ptr copy_async(typename I::ptr dst, device_input &input, size_t count, typename I::stream &stream, list_event<I> &dependencies, gmacError_t &err);
    typename I::event_ptr copy_async(typename I::ptr dst, device_input &input, size_t count, typename I::stream &stream, typename I::event_ptr event, gmacError_t &err);
    typename I::event_ptr copy_async(typename I::ptr dst, device_input &input, size_t count, typename I::stream &stream, gmacError_t &err);

    typename I::event_ptr copy_async(device_output &output, typename I::ptr_const src, size_t count, typename I::stream &stream, list_event<I> &dependencies, gmacError_t &err);
    typename I::event_ptr copy_async(device_output &output, typename I::ptr_const src, size_t count, typename I::stream &stream, typename I::event_ptr event, gmacError_t &err);
    typename I::event_ptr copy_async(device_output &output, typename I::ptr_const src, size_t count, typename I::stream &stream, gmacError_t &err);

    typename I::event_ptr memset(typename I::ptr dst, int c, size_t count, typename I::stream &stream, list_event<I> &dependencies, gmacError_t &err);
    typename I::event_ptr memset(typename I::ptr dst, int c, size_t count, typename I::stream &stream, typename I::event_ptr event, gmacError_t &err);
    typename I::event_ptr memset(typename I::ptr dst, int c, size_t count, typename I::stream &stream, gmacError_t &err);

    typename I::event_ptr memset_async(typename I::ptr dst, int c, size_t count, typename I::stream &stream, list_event<I> &dependencies, gmacError_t &err);
    typename I::event_ptr memset_async(typename I::ptr dst, int c, size_t count, typename I::stream &stream, typename I::event_ptr event, gmacError_t &err);
    typename I::event_ptr memset_async(typename I::ptr dst, int c, size_t count, typename I::stream &stream, gmacError_t &err);

    virtual typename I::code_repository &get_code_repository() = 0;
};

template <typename D, typename B, typename I>
const unsigned &context_t<D, B, I>::MaxBuffersIn_  = config::params::HALInputBuffersPerContext;

template <typename D, typename B, typename I>
const unsigned &context_t<D, B, I>::MaxBuffersOut_ = config::params::HALOutputBuffersPerContext;

}

}}

#endif /* ASPACE_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
