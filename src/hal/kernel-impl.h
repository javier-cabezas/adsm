#ifndef GMAC_HAL_KERNEL_IMPL_H_
#define GMAC_HAL_KERNEL_IMPL_H_

namespace __impl { namespace hal { namespace detail {

template <typename I>
inline
kernel_t<I>::kernel_t(const std::string &name) :
    name_(name)
{
}

template <typename I>
inline
const std::string &
kernel_t<I>::get_name() const
{
    return name_;
}

template <typename I>
inline
kernel_t<I>::config::config(unsigned ndims) :
    ndims_(ndims)
{
}

template <typename I>
inline
unsigned
kernel_t<I>::config::get_ndims() const
{
    return ndims_;
}

template <typename I>
inline
kernel_t<I>::launch::launch(typename I::kernel &kernel,
                               typename I::kernel::config &config,
                               typename I::kernel::arg_list &args, 
                               typename I::stream &stream) :

    kernel_(kernel),
    config_(config),
    args_(args),
    stream_(stream)
{
}

template <typename I>
inline
typename I::stream &
kernel_t<I>::launch::get_stream()
{
    return stream_;
}

template <typename I>
inline
const typename I::stream &
kernel_t<I>::launch::get_stream() const
{
    return stream_;
}

template <typename I>
inline
const typename I::kernel &
kernel_t<I>::launch::get_kernel() const
{
    return reinterpret_cast<typename I::kernel &>(kernel_);
}

template <typename I>
inline
const typename I::kernel::config &
kernel_t<I>::launch::get_config() const
{
    return config_;
}

template <typename I>
inline
const typename I::kernel::arg_list &
kernel_t<I>::launch::get_arg_list() const
{
    return args_;
}

template <typename I>
inline
typename I::event_ptr
kernel_t<I>::launch::get_event()
{
    return event_;
}

} // namespace detail

}}

#endif /* GMAC_HAL_KERNEL_IMPL_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
