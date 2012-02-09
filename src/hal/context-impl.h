#ifndef GMAC_HAL_CONTEXT_IMPL_H_
#define GMAC_HAL_CONTEXT_IMPL_H_

namespace __impl { namespace hal { namespace detail {

template <typename I>
inline
buffer_t<I>::buffer_t(size_t size, typename I::context &context) :
    context_(context),
    size_(size)
{
}

template <typename I>
inline
typename I::context &
buffer_t<I>::get_context()
{
    return context_;
}

template <typename I>
inline
const typename I::context &
buffer_t<I>::get_context() const
{
    return context_;
}

template <typename I>
inline
size_t
buffer_t<I>::get_size() const
{
    return size_;
}

template <typename I>
inline
queue_event<I>::queue_event() :
    Lock("queue_event")
{
}

template <typename I>
typename I::event *
queue_event<I>::pop()
{
    typename I::event *ret = NULL;

    Lock::lock();
    if (Parent::size() > 0) {
        ret = Parent::front();
        Parent::pop();
    }
    Lock::unlock();

    return ret;
}

template <typename I>
void
queue_event<I>::push(typename I::event &event)
{
	Lock::lock();
    Parent::push(&event);
    Lock::unlock();
}

template <typename I>
host_ptr
context_t<I>::get_memory(size_t size)
{
    host_ptr mem = (host_ptr) mapMemory_.pop(size);

    if (mem == NULL) {
        mem = (host_ptr) malloc(size);
    }

    return mem;
}

template <typename I>
inline void
context_t<I>::put_memory(void *ptr, size_t size)
{
    mapMemory_.push(ptr, size);
}

template <typename I>
inline
context_t<I>::context_t(I_device &dev) :
    device_(dev),
    nBuffersIn_(0),
    nBuffersOut_(0)
{
}

template <typename I>
inline
typename context_t<I>::I_device &
context_t<I>::get_device()
{
    return device_;
}

template <typename I>
inline
const typename context_t<I>::I_device &
context_t<I>::get_device() const
{
    return device_;
}


template <typename I>
typename I::event_ptr 
context_t<I>::copy(typename I::ptr dst, typename I::ptr_const src, size_t count, typename I::stream &stream, list_event<I> &_dependencies, gmacError_t &err)
{
    typename I::event_ptr ret = copy_backend(dst, src, count, stream, &_dependencies, err);
    if (err == gmacSuccess) {
        _dependencies.set_synced();
    }

    return ret;
}

template <typename I>
typename I::event_ptr 
context_t<I>::copy(typename I::ptr dst, typename I::ptr_const src, size_t count, typename I::stream &stream, typename I::event_ptr event, gmacError_t &err)
{
    typename I::event_list list;
    list.add_event(event);

    typename I::event_ptr ret = copy_backend(dst, src, count, stream, &list, err);
    if (err == gmacSuccess) {
        event->set_synced();
    }

    return ret;
}

template <typename I>
typename I::event_ptr 
context_t<I>::copy(typename I::ptr dst, typename I::ptr_const src, size_t count, typename I::stream &stream, gmacError_t &err)
{
    return copy_backend(dst, src, count, stream, NULL, err);
}

template <typename I>
typename I::event_ptr
context_t<I>::copy(typename I::ptr dst, device_input &input, size_t count, typename I::stream &stream, list_event<I> &_dependencies, gmacError_t &err)
{
    typename I::event_ptr ret = copy_backend(dst, input, count, stream, &_dependencies, err);
    if (err == gmacSuccess) {
        _dependencies.set_synced();
    }

    return ret;
}

template <typename I>
typename I::event_ptr
context_t<I>::copy(typename I::ptr dst, device_input &input, size_t count, typename I::stream &stream, typename I::event_ptr event, gmacError_t &err)
{
    typename I::event_list list;
    list.add_event(event);

    typename I::event_ptr ret = copy_backend(dst, input, count, stream, &list, err);
    if (err == gmacSuccess) {
        event->set_synced();
    }
}

template <typename I>
typename I::event_ptr
context_t<I>::copy(typename I::ptr dst, device_input &input, size_t count, typename I::stream &stream, gmacError_t &err)
{
    return copy_backend(dst, input, count, stream, NULL, err);
}

template <typename I>
typename I::event_ptr
context_t<I>::copy(device_output &output, typename I::ptr_const src, size_t count, typename I::stream &stream, list_event<I> &_dependencies, gmacError_t &err)
{
    typename I::event_ptr ret = copy_backend(output, src, count, stream, &_dependencies, err);
    if (err == gmacSuccess) {
        _dependencies.set_synced();
    }

    return ret;
}

template <typename I>
typename I::event_ptr
context_t<I>::copy(device_output &output, typename I::ptr_const src, size_t count, typename I::stream &stream, typename I::event_ptr event, gmacError_t &err)
{
    typename I::event_list list;
    list.add_event(event);

    typename I::event_ptr ret = copy_backend(output, src, count, stream, &list, err);
    if (err == gmacSuccess) {
        event->set_synced();
    }

    return ret;
}

template <typename I>
typename I::event_ptr
context_t<I>::copy(device_output &output, typename I::ptr_const src, size_t count, typename I::stream &stream, gmacError_t &err)
{
    return copy_backend(output, src, count, stream, NULL, err);
}

template <typename I>
typename I::event_ptr 
context_t<I>::copy_async(typename I::ptr dst, typename I::ptr_const src, size_t count, typename I::stream &stream, list_event<I> &_dependencies, gmacError_t &err)
{
    typename I::event_ptr ret = copy_async_backend(dst, src, count, stream, &_dependencies, err);
    if (err == gmacSuccess) {
        _dependencies.set_synced();
    }

    return ret;
}

template <typename I>
typename I::event_ptr 
context_t<I>::copy_async(typename I::ptr dst, typename I::ptr_const src, size_t count, typename I::stream &stream, typename I::event_ptr event, gmacError_t &err)
{
    typename I::event_list list;
    list.add_event(event);

    typename I::event_ptr ret = copy_async_backend(dst, src, count, stream, &list, err);
    if (err == gmacSuccess) {
        event->set_synced();
    }

    return ret;
}

template <typename I>
typename I::event_ptr 
context_t<I>::copy_async(typename I::ptr dst, typename I::ptr_const src, size_t count, typename I::stream &stream, gmacError_t &err)
{
    return copy_async_backend(dst, src, count, stream, NULL, err);
}

template <typename I>
typename I::event_ptr
context_t<I>::copy_async(typename I::ptr dst, device_input &input, size_t count, typename I::stream &stream, list_event<I> &_dependencies, gmacError_t &err)
{
    typename I::event_ptr ret = copy_async_backend(dst, input, count, stream, &_dependencies, err);
    if (err == gmacSuccess) {
        _dependencies.set_synced();
    }

    return ret;
}

template <typename I>
typename I::event_ptr
context_t<I>::copy_async(typename I::ptr dst, device_input &input, size_t count, typename I::stream &stream, typename I::event_ptr event, gmacError_t &err)
{
    typename I::event_list list;
    list.add_event(event);

    typename I::event_ptr ret = copy_async_backend(dst, input, count, stream, &list, err);
    if (err == gmacSuccess) {
        event->set_synced();
    }

    return ret;
}

template <typename I>
typename I::event_ptr
context_t<I>::copy_async(typename I::ptr dst, device_input &input, size_t count, typename I::stream &stream, gmacError_t &err)
{
    return copy_async_backend(dst, input, count, stream, NULL, err);
}

template <typename I>
typename I::event_ptr
context_t<I>::copy_async(device_output &output, typename I::ptr_const src, size_t count, typename I::stream &stream, list_event<I> &_dependencies, gmacError_t &err)
{
    typename I::event_ptr ret = copy_async_backend(output, src, count, stream, &_dependencies, err);
    if (err == gmacSuccess) {
        _dependencies.set_synced();
    }

    return ret;
}

template <typename I>
typename I::event_ptr
context_t<I>::copy_async(device_output &output, typename I::ptr_const src, size_t count, typename I::stream &stream, typename I::event_ptr event, gmacError_t &err)
{
    typename I::event_list list;
    list.add_event(event);

    typename I::event_ptr ret = copy_async_backend(output, src, count, stream, &list, err);
    if (err == gmacSuccess) {
        event->set_synced();
    }

    return ret;
}

template <typename I>
typename I::event_ptr
context_t<I>::copy_async(device_output &output, typename I::ptr_const src, size_t count, typename I::stream &stream, gmacError_t &err)
{
    return copy_async_backend(output, src, count, stream, NULL, err);
}

template <typename I>
typename I::event_ptr 
context_t<I>::memset(typename I::ptr dst, int c, size_t count, typename I::stream &stream, list_event<I> &_dependencies, gmacError_t &err)
{
    typename I::event_ptr ret = memset_backend(dst, c, count, stream, &_dependencies, err);
    if (err == gmacSuccess) {
        _dependencies.set_synced();
    }

    return ret;
}

template <typename I>
typename I::event_ptr 
context_t<I>::memset(typename I::ptr dst, int c, size_t count, typename I::stream &stream, typename I::event_ptr event, gmacError_t &err)
{
    typename I::event_list list;
    list.add_event(event);

    typename I::event_ptr ret = memset_backend(dst, c, count, stream, &list, err);
    if (err == gmacSuccess) {
        event->set_synced();
    }

    return ret;
}

template <typename I>
typename I::event_ptr 
context_t<I>::memset(typename I::ptr dst, int c, size_t count, typename I::stream &stream, gmacError_t &err)
{
    return memset_backend(dst, c, count, stream, NULL, err);
}

template <typename I>
typename I::event_ptr 
context_t<I>::memset_async(typename I::ptr dst, int c, size_t count, typename I::stream &stream, list_event<I> &_dependencies, gmacError_t &err)
{
    typename I::event_ptr ret = memset_async_backend(dst, c, count, stream, &_dependencies, err);
    if (err == gmacSuccess) {
        _dependencies.set_synced();
    }

    return ret;
}

template <typename I>
typename I::event_ptr 
context_t<I>::memset_async(typename I::ptr dst, int c, size_t count, typename I::stream &stream, typename I::event_ptr event, gmacError_t &err)
{
    typename I::event_list list;
    list.add_event(event);

    typename I::event_ptr ret = memset_async_backend(dst, c, count, stream, &list, err);
    if (err == gmacSuccess) {
        event->set_synced();
    }

    return ret;
}

template <typename I>
typename I::event_ptr 
context_t<I>::memset_async(typename I::ptr dst, int c, size_t count, typename I::stream &stream, gmacError_t &err)
{
    return memset_async_backend(dst, c, count, stream, NULL, err);
}

template <typename I>
typename I::buffer *
context_t<I>::get_input_buffer(size_t size, typename I::stream &stream, typename I::event_ptr event)
{
    typename I::buffer *buffer = mapFreeBuffersIn_.pop(size);

    if (buffer == NULL) {
        nBuffersIn_.lock();
        if (nBuffersIn_ < MaxBuffersIn_) {
            nBuffersIn_++;
            nBuffersIn_.unlock();

            gmacError_t err;
            buffer = alloc_buffer(size, GMAC_PROT_READ, stream, err);
            ASSERTION(err == gmacSuccess);
        } else {
            nBuffersIn_.unlock();
            buffer = mapUsedBuffersIn_.pop(size);
            buffer->wait();
        }
    } else {
        TRACE(LOCAL, "Reusing input buffer");
    }

    buffer->set_event(event);

    mapUsedBuffersIn_.push(buffer, buffer->get_size());

    return buffer;
}

template <typename I>
typename I::buffer *
context_t<I>::get_output_buffer(size_t size, typename I::stream &stream, typename I::event_ptr event)
{
    typename I::buffer *buffer = mapFreeBuffersOut_.pop(size);

    if (buffer == NULL) {
        if (nBuffersOut_ < MaxBuffersOut_) {
            gmacError_t err;

            buffer = alloc_buffer(size, GMAC_PROT_WRITE, stream, err);
            ASSERTION(err == gmacSuccess);
            nBuffersOut_++;
        } else {
            buffer = mapUsedBuffersOut_.pop(size);
            buffer->wait();
        }
    } else {
        TRACE(LOCAL, "Reusing output buffer");
    }

    buffer->set_event(event);

    mapUsedBuffersOut_.push(buffer, buffer->get_size());

    return buffer;
}

template <typename I>
void
context_t<I>::put_input_buffer(typename I::buffer &buffer)
{
    mapUsedBuffersIn_.remove(&buffer, buffer.get_size());
    mapFreeBuffersIn_.push(&buffer, buffer.get_size());
}

template <typename I>
void
context_t<I>::put_output_buffer(typename I::buffer &buffer)
{
    mapUsedBuffersOut_.remove(&buffer, buffer.get_size());
    mapFreeBuffersOut_.push(&buffer, buffer.get_size());
}

} // namespace detail

}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
