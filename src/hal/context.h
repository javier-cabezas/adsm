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

template <typename T>
class GMAC_LOCAL map_pool :
    std::map<size_t, std::list<T *> >,
    gmac::util::mutex<map_pool<T> > {

    typedef std::list<T *> queue_subset;
    typedef std::map<size_t, queue_subset> Parent;

public:
    map_pool() :
        gmac::util::mutex<map_pool>("map_pool")
    {
    }

    T *pop(size_t size)
    {
        T *ret = NULL;

        this->lock();
        typename Parent::iterator it;
        it = Parent::lower_bound(size);
        if (it != Parent::end()) {
            queue_subset &queue = it->second;
            if (queue.size() > 0) {
                ret = queue.front();
                queue.pop_front();
            }
        }
        this->unlock();

        return ret;
    }

    bool remove(T *val, size_t size)
    {
        bool ret = false;

        this->lock();
        typename Parent::iterator it;
        it = Parent::lower_bound(size);
        if (it != Parent::end()) {
            queue_subset &queue = it->second;

            typename queue_subset::iterator it2;
            it2 = std::find(queue.begin(), queue.end(), val);
            if (it2 != queue.end()) {
                queue.erase(it2);
                ret = true;
            }
        }
        this->unlock();

        return ret;
    }

    void push(T *v, size_t size)
    {
    	this->lock();
        typename Parent::iterator it;
        it = Parent::find(size);
        if (it != Parent::end()) {
            queue_subset &queue = it->second;
            queue.push_back(v);
        } else {
            queue_subset queue;
            queue.push_back(v);
            Parent::insert(typename Parent::value_type(size, queue));
        }
        this->unlock();
    }
};



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

template <typename I>
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

template <typename I>
class GMAC_LOCAL context_t :
    public util::attributes<context_t<I> >,
    public util::on_construction<typename I::context>,
    public util::on_destruction<typename I::context> {

private:
    typedef map_pool<void> map_memory;
    typedef map_pool<typename I::buffer> map_buffer;

protected:
    typedef util::locked_counter<unsigned, gmac::util::spinlock<context_t> > buffer_counter;

    typedef typename I::ptr I_ptr;
    typedef typename I::ptr_const I_ptr_const;
    typedef typename I::event_ptr I_event_ptr;
    typedef typename I::stream I_stream;
    typedef typename I::device I_device;

    I_device &device_;

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

    typename I::buffer *get_input_buffer(size_t size, I_stream &stream, I_event_ptr event);
    void put_input_buffer(typename I::buffer &buffer);
 
    typename I::buffer *get_output_buffer(size_t size, I_stream &stream, I_event_ptr event);
    void put_output_buffer(typename I::buffer &buffer);

    virtual typename I::buffer *alloc_buffer(size_t size, GmacProtection hint, I_stream &stream, gmacError_t &err) = 0;
    virtual gmacError_t free_buffer(typename I::buffer &buffer) = 0;

    context_t(I_device &device);

    virtual I_event_ptr copy_backend(I_ptr dst, I_ptr_const src, size_t count, I_stream &stream, list_event<I> *dependencies, gmacError_t &err) = 0;
    virtual I_event_ptr copy_backend(I_ptr dst, device_input &input, size_t count, I_stream &stream, list_event<I> *dependencies, gmacError_t &err) = 0;
    virtual I_event_ptr copy_backend(device_output &output, I_ptr_const src, size_t count, I_stream &stream, list_event<I> *dependencies, gmacError_t &err) = 0;
    virtual I_event_ptr memset_backend(I_ptr dst, int c, size_t count, I_stream &stream, list_event<I> *dependencies, gmacError_t &err) = 0;

    virtual I_event_ptr copy_async_backend(I_ptr dst, I_ptr_const src, size_t count, I_stream &stream, list_event<I> *dependencies, gmacError_t &err) = 0;
    virtual I_event_ptr copy_async_backend(I_ptr dst, device_input &input, size_t count, I_stream &stream, list_event<I> *dependencies, gmacError_t &err) = 0;
    virtual I_event_ptr copy_async_backend(device_output &output, I_ptr_const src, size_t count, I_stream &stream, list_event<I> *dependencies, gmacError_t &err) = 0;
    virtual I_event_ptr memset_async_backend(I_ptr dst, int c, size_t count, I_stream &stream, list_event<I> *dependencies, gmacError_t &err) = 0;

public:
    I_device &get_device();
    const I_device &get_device() const;

    virtual I_ptr alloc(size_t size, gmacError_t &err) = 0;
    virtual I_ptr alloc_host_pinned(size_t size, GmacProtection hint, gmacError_t &err) = 0;

    virtual gmacError_t free(I_ptr acc) = 0;
    virtual gmacError_t free_host_pinned(I_ptr I_ptr) = 0;

    I_event_ptr copy(I_ptr dst, I_ptr_const src, size_t count, I_stream &stream, list_event<I> &dependencies, gmacError_t &err);
    I_event_ptr copy(I_ptr dst, I_ptr_const src, size_t count, I_stream &stream, I_event_ptr event, gmacError_t &err);
    I_event_ptr copy(I_ptr dst, I_ptr_const src, size_t count, I_stream &stream, gmacError_t &err);

    I_event_ptr copy(I_ptr dst, device_input &input, size_t count, I_stream &stream, list_event<I> &dependencies, gmacError_t &err);
    I_event_ptr copy(I_ptr dst, device_input &input, size_t count, I_stream &stream, I_event_ptr event, gmacError_t &err);
    I_event_ptr copy(I_ptr dst, device_input &input, size_t count, I_stream &stream, gmacError_t &err);

    I_event_ptr copy(device_output &output, I_ptr_const src, size_t count, I_stream &stream, list_event<I> &dependencies, gmacError_t &err);
    I_event_ptr copy(device_output &output, I_ptr_const src, size_t count, I_stream &stream, I_event_ptr event, gmacError_t &err);
    I_event_ptr copy(device_output &output, I_ptr_const src, size_t count, I_stream &stream, gmacError_t &err);

    I_event_ptr copy_async(I_ptr dst, I_ptr_const src, size_t count, I_stream &stream, list_event<I> &dependencies, gmacError_t &err);
    I_event_ptr copy_async(I_ptr dst, I_ptr_const src, size_t count, I_stream &stream, I_event_ptr event, gmacError_t &err);
    I_event_ptr copy_async(I_ptr dst, I_ptr_const src, size_t count, I_stream &stream, gmacError_t &err);

    I_event_ptr copy_async(I_ptr dst, device_input &input, size_t count, I_stream &stream, list_event<I> &dependencies, gmacError_t &err);
    I_event_ptr copy_async(I_ptr dst, device_input &input, size_t count, I_stream &stream, I_event_ptr event, gmacError_t &err);
    I_event_ptr copy_async(I_ptr dst, device_input &input, size_t count, I_stream &stream, gmacError_t &err);

    I_event_ptr copy_async(device_output &output, I_ptr_const src, size_t count, I_stream &stream, list_event<I> &dependencies, gmacError_t &err);
    I_event_ptr copy_async(device_output &output, I_ptr_const src, size_t count, I_stream &stream, I_event_ptr event, gmacError_t &err);
    I_event_ptr copy_async(device_output &output, I_ptr_const src, size_t count, I_stream &stream, gmacError_t &err);

    I_event_ptr memset(I_ptr dst, int c, size_t count, I_stream &stream, list_event<I> &dependencies, gmacError_t &err);
    I_event_ptr memset(I_ptr dst, int c, size_t count, I_stream &stream, I_event_ptr event, gmacError_t &err);
    I_event_ptr memset(I_ptr dst, int c, size_t count, I_stream &stream, gmacError_t &err);

    I_event_ptr memset_async(I_ptr dst, int c, size_t count, I_stream &stream, list_event<I> &dependencies, gmacError_t &err);
    I_event_ptr memset_async(I_ptr dst, int c, size_t count, I_stream &stream, I_event_ptr event, gmacError_t &err);
    I_event_ptr memset_async(I_ptr dst, int c, size_t count, I_stream &stream, gmacError_t &err);

    virtual typename I::code_repository &get_code_repository() = 0;
};

template <typename I>
const unsigned &context_t<I>::MaxBuffersIn_  = config::params::HALInputBuffersPerContext;

template <typename I>
const unsigned &context_t<I>::MaxBuffersOut_ = config::params::HALOutputBuffersPerContext;

}

}}

#endif /* ASPACE_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
