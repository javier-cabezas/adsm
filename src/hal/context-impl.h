#ifndef GMAC_HAL_CONTEXT_IMPL_H_
#define GMAC_HAL_CONTEXT_IMPL_H_

namespace __impl { namespace hal { namespace detail {

template <typename D, typename B, typename I>
inline
buffer_t<D, B, I>::buffer_t(typename I::context &context) :
    context_(context)
{
}

template <typename D, typename B, typename I>
inline
typename I::context &
buffer_t<D, B, I>::get_context()
{
    return context_;
}

template <typename D, typename B, typename I>
inline
const typename I::context &
buffer_t<D, B, I>::get_context() const
{
    return context_;
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

} // namespace detail

}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
