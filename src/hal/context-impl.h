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
void
list_event<I>::add_event(typename I::event event) 
{
    Parent::push_back(event);
}

template <typename I>
inline
queue_event<I>::queue_event() :
    gmac::util::mutex("queue_event")
{
}

template <typename I>
typename I::event::event_type *
queue_event<I>::pop()
{
    typename I::event::event_type *ret = NULL;

    lock();
    if (Parent::size() > 0) {
        ret = Parent::front();
        Parent::pop();
    }
    unlock();

    return ret;
}

template <typename I>
void
queue_event<I>::push(typename I::event::event_type &event)
{
    lock();
    Parent::push(&event);
    unlock();
}

template <typename D, typename B, typename I>
inline
context_t<D, B, I>::context_t(typename B::context context, D &dev) :
    context_(context),
    device_(dev)
{
}

template <typename D, typename B, typename I>
inline
D &
context_t<D, B, I>::get_device()
{
    return device_;
}

template <typename D, typename B, typename I>
inline
const D &
context_t<D, B, I>::get_device() const
{
    return device_;
}


template <typename D, typename B, typename I>
inline
typename B::context &
context_t<D, B, I>::operator()()
{
    return context_;
}

template <typename D, typename B, typename I>
inline
const typename B::context &
context_t<D, B, I>::operator()() const
{
    return context_;
}

template <typename D, typename B, typename I>
typename I::event 
context_t<D, B, I>::copy(typename I::ptr dst, const typename I::ptr src, size_t count, typename I::stream &stream, list_event<I> &_dependencies, gmacError_t &err)
{
    return copy_backend(dst, src, count, stream, &_dependencies, err);
}

template <typename D, typename B, typename I>
typename I::event 
context_t<D, B, I>::copy(typename I::ptr dst, const typename I::ptr src, size_t count, typename I::stream &stream, typename I::event event, gmacError_t &err)
{
    typename I::event_list list;
    list.add_event(event);

    return copy_backend(dst, src, count, stream, &list, err);
}

template <typename D, typename B, typename I>
typename I::event 
context_t<D, B, I>::copy(typename I::ptr dst, const typename I::ptr src, size_t count, typename I::stream &stream, gmacError_t &err)
{
    return copy_backend(dst, src, count, stream, NULL, err);
}

template <typename D, typename B, typename I>
typename I::event
context_t<D, B, I>::copy(typename I::ptr dst, device_input &input, size_t count, typename I::stream &stream, list_event<I> &_dependencies, gmacError_t &err)
{
    return copy_backend(dst, input, count, stream, &_dependencies, err);
}

template <typename D, typename B, typename I>
typename I::event
context_t<D, B, I>::copy(typename I::ptr dst, device_input &input, size_t count, typename I::stream &stream, typename I::event event, gmacError_t &err)
{
    typename I::event_list list;
    list.add_event(event);

    return copy_backend(dst, input, count, stream, &list, err);
}

template <typename D, typename B, typename I>
typename I::event
context_t<D, B, I>::copy(typename I::ptr dst, device_input &input, size_t count, typename I::stream &stream, gmacError_t &err)
{
    return copy_backend(dst, input, count, stream, NULL, err);
}

template <typename D, typename B, typename I>
typename I::event
context_t<D, B, I>::copy(device_output &output, const typename I::ptr src, size_t count, typename I::stream &stream, list_event<I> &_dependencies, gmacError_t &err)
{
    return copy_backend(output, src, count, stream, &_dependencies, err);
}

template <typename D, typename B, typename I>
typename I::event
context_t<D, B, I>::copy(device_output &output, const typename I::ptr src, size_t count, typename I::stream &stream, typename I::event event, gmacError_t &err)
{
    typename I::event_list list;
    list.add_event(event);

    return copy_backend(output, src, count, stream, &list, err);
}

template <typename D, typename B, typename I>
typename I::event
context_t<D, B, I>::copy(device_output &output, const typename I::ptr src, size_t count, typename I::stream &stream, gmacError_t &err)
{
    return copy_backend(output, src, count, stream, NULL, err);
}

template <typename D, typename B, typename I>
typename I::event 
context_t<D, B, I>::copy_async(typename I::ptr dst, const typename I::ptr src, size_t count, typename I::stream &stream, list_event<I> &_dependencies, gmacError_t &err)
{
    return copy_async_backend(dst, src, count, stream, &_dependencies, err);
}

template <typename D, typename B, typename I>
typename I::event 
context_t<D, B, I>::copy_async(typename I::ptr dst, const typename I::ptr src, size_t count, typename I::stream &stream, typename I::event event, gmacError_t &err)
{
    typename I::event_list list;
    list.add_event(event);

    return copy_async_backend(dst, src, count, stream, &list, err);
}

template <typename D, typename B, typename I>
typename I::event 
context_t<D, B, I>::copy_async(typename I::ptr dst, const typename I::ptr src, size_t count, typename I::stream &stream, gmacError_t &err)
{
    return copy_async_backend(dst, src, count, stream, NULL, err);
}

template <typename D, typename B, typename I>
typename I::event
context_t<D, B, I>::copy_async(typename I::ptr dst, device_input &input, size_t count, typename I::stream &stream, list_event<I> &_dependencies, gmacError_t &err)
{
    return copy_async_backend(dst, input, count, stream, &_dependencies, err);
}

template <typename D, typename B, typename I>
typename I::event
context_t<D, B, I>::copy_async(typename I::ptr dst, device_input &input, size_t count, typename I::stream &stream, typename I::event event, gmacError_t &err)
{
    typename I::event_list list;
    list.add_event(event);

    return copy_async_backend(dst, input, count, stream, &list, err);
}

template <typename D, typename B, typename I>
typename I::event
context_t<D, B, I>::copy_async(typename I::ptr dst, device_input &input, size_t count, typename I::stream &stream, gmacError_t &err)
{
    return copy_async_backend(dst, input, count, stream, NULL, err);
}

template <typename D, typename B, typename I>
typename I::event
context_t<D, B, I>::copy_async(device_output &output, const typename I::ptr src, size_t count, typename I::stream &stream, list_event<I> &_dependencies, gmacError_t &err)
{
    return copy_async_backend(output, src, count, stream, &_dependencies, err);
}

template <typename D, typename B, typename I>
typename I::event
context_t<D, B, I>::copy_async(device_output &output, const typename I::ptr src, size_t count, typename I::stream &stream, typename I::event event, gmacError_t &err)
{
    typename I::event_list list;
    list.add_event(event);

    return copy_async_backend(output, src, count, stream, &list, err);
}

template <typename D, typename B, typename I>
typename I::event
context_t<D, B, I>::copy_async(device_output &output, const typename I::ptr src, size_t count, typename I::stream &stream, gmacError_t &err)
{
    return copy_async_backend(output, src, count, stream, NULL, err);
}

template <typename D, typename B, typename I>
typename I::event 
context_t<D, B, I>::memset(typename I::ptr dst, int c, size_t count, typename I::stream &stream, list_event<I> &_dependencies, gmacError_t &err)
{
    return memset_backend(dst, c, count, stream, &_dependencies, err);
}

template <typename D, typename B, typename I>
typename I::event 
context_t<D, B, I>::memset(typename I::ptr dst, int c, size_t count, typename I::stream &stream, typename I::event event, gmacError_t &err)
{
    typename I::event_list list;
    list.add_event(event);

    return memset_backend(dst, c, count, stream, &list, err);
}

template <typename D, typename B, typename I>
typename I::event 
context_t<D, B, I>::memset(typename I::ptr dst, int c, size_t count, typename I::stream &stream, gmacError_t &err)
{
    return memset_backend(dst, c, count, stream, NULL, err);
}

template <typename D, typename B, typename I>
typename I::event 
context_t<D, B, I>::memset_async(typename I::ptr dst, int c, size_t count, typename I::stream &stream, list_event<I> &_dependencies, gmacError_t &err)
{
    return memset_async_backend(dst, c, count, stream, &_dependencies, err);
}

template <typename D, typename B, typename I>
typename I::event 
context_t<D, B, I>::memset_async(typename I::ptr dst, int c, size_t count, typename I::stream &stream, typename I::event event, gmacError_t &err)
{
    typename I::event_list list;
    list.add_event(event);

    return memset_async_backend(dst, c, count, stream, &list, err);
}

template <typename D, typename B, typename I>
typename I::event 
context_t<D, B, I>::memset_async(typename I::ptr dst, int c, size_t count, typename I::stream &stream, gmacError_t &err)
{
    return memset_async_backend(dst, c, count, stream, NULL, err);
}

} // namespace detail

}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
