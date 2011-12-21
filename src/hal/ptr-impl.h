#ifndef GMAC_HAL_PTR_IMPL_H_
#define GMAC_HAL_PTR_IMPL_H_

namespace __impl { namespace hal {

template <typename HPtr, typename Ptr, typename C>
inline
_ptr_t<HPtr, Ptr, C>::_ptr_t(Ptr ptr, C *ctx) :
    ptrDev_(ptr),
    ctx_(ctx)
{
}

template <typename HPtr, typename Ptr, typename C>
inline
_ptr_t<HPtr, Ptr, C>::_ptr_t(typename Ptr::backend_type value, C *ctx) :
    ptrDev_(value),
    ctx_(ctx)
{
}

template <typename HPtr, typename Ptr, typename C>
inline
_ptr_t<HPtr, Ptr, C>::_ptr_t(host_ptr ptr) :
    ptrDev_(0),
    ptrHost_(ptr),
    ctx_(NULL)
{
}

template <typename HPtr, typename Ptr, typename C>
inline
_ptr_t<HPtr, Ptr, C>::_ptr_t(host_const_ptr ptr) :
    ptrDev_(0),
    ptrHost_(ptr),
    ctx_(NULL)
{
}

template <typename HPtr, typename Ptr, typename C>
inline
_ptr_t<HPtr, Ptr, C>::_ptr_t() :
    ptrDev_(0),
    ptrHost_(0),
    ctx_(NULL)
{
}

template <typename HPtr, typename Ptr, typename C>
inline
_ptr_t<HPtr, Ptr, C>::_ptr_t(const _ptr_t<host_ptr, Ptr, C> &ptr) :
    ptrDev_(ptr.ptrDev_),
    ptrHost_(ptr.ptrHost_),
    ctx_(ptr.ctx_)
{
}

template <typename HPtr, typename Ptr, typename C>
inline
_ptr_t<HPtr, Ptr, C>::_ptr_t(const _ptr_t<host_const_ptr, Ptr, C> &ptr) :
    ptrDev_(ptr.ptrDev_),
    ptrHost_(ptr.ptrHost_),
    ctx_(ptr.ctx_)
{
}

template <typename HPtr, typename Ptr, typename C>
inline
_ptr_t<HPtr, Ptr, C>::operator bool() const
{
    return ctx_ != NULL || ptrHost_ != NULL;
}

template <typename HPtr, typename Ptr, typename C>
inline _ptr_t<HPtr, Ptr, C> &
_ptr_t<HPtr, Ptr, C>::operator=(const _ptr_t &ptr)
{
    if (this != &ptr) {
        ptrDev_ = ptr.ptrDev_;
        ptrHost_ = ptr.ptrHost_;
        ctx_  = ptr.ctx_;
    }
    return *this;
}

template <typename HPtr, typename Ptr, typename C>
inline bool
_ptr_t<HPtr, Ptr, C>::operator==(const _ptr_t &ptr) const
{
    bool ret;
    if (ctx_ == NULL) {
        ret = (ptrHost_ == ptr.ptrHost_);
    } else {
        ret = (ptrDev_ == ptr.ptrDev_);
    }
    return ret;
}

template <typename HPtr, typename Ptr, typename C>
inline bool
_ptr_t<HPtr, Ptr, C>::operator==(long i) const
{
    bool ret;
    if (ctx_ == NULL) {
        ret = (ptrHost_ == HPtr(i));
    } else {
        ret = (ptrDev_ == i);
    }
    return ret;
}

template <typename HPtr, typename Ptr, typename C>
inline bool
_ptr_t<HPtr, Ptr, C>::operator!=(const _ptr_t &ptr) const
{
    bool ret;
    if (ctx_ == NULL) {
        ret = (ptrHost_ != ptr.ptrHost_);
    } else {
        ret = (ptrDev_ != ptr.ptrDev_);
    }
    return ret;
}

template <typename HPtr, typename Ptr, typename C>
inline bool
_ptr_t<HPtr, Ptr, C>::operator!=(long i) const
{
    bool ret;
    if (ctx_ == NULL) {
        ret = (ptrHost_ != HPtr(i));
    } else {
        ret = (ptrDev_ != i);
    }
    return ret;

}

template <typename HPtr, typename Ptr, typename C>
inline bool
_ptr_t<HPtr, Ptr, C>::operator<(const _ptr_t &ptr) const
{
    bool ret;
    if (ctx_ == NULL) {
        return ptrHost_ < ptr.ptrHost_;
    } else {
        return ctx_ < ptr.ctx_ || (ctx_ == ptr.ctx_ && ptrDev_ < ptr.ptrDev_);
    }
    return ret;
}

template <typename HPtr, typename Ptr, typename C>
template <typename T>
inline _ptr_t<HPtr, Ptr, C> &
_ptr_t<HPtr, Ptr, C>::operator+=(const T &off)
{
    if (ctx_ == NULL) {
        ptrHost_ += off;
    } else {
        ptrDev_ += off;
    }
    return *this;
}

template <typename HPtr, typename Ptr, typename C>
template <typename T>
inline const _ptr_t<HPtr, Ptr, C>
_ptr_t<HPtr, Ptr, C>::operator+(const T &off) const
{
    _ptr_t ret(*this);
    ret += off;
    return ret;
}

template <typename HPtr, typename Ptr, typename C>
inline typename Ptr::backend_type
_ptr_t<HPtr, Ptr, C>::get_device_addr() const
{
    ASSERTION(is_device_ptr());
    return ptrDev_.get();
}

template <typename HPtr, typename Ptr, typename C>
inline HPtr
_ptr_t<HPtr, Ptr, C>::get_host_addr() const
{
    ASSERTION(is_host_ptr());
    return ptrHost_;
}

template <typename HPtr, typename Ptr, typename C>
inline size_t
_ptr_t<HPtr, Ptr, C>::get_offset() const
{
    ASSERTION(is_device_ptr());
    return ptrDev_.offset();
}

template <typename HPtr, typename Ptr, typename C>
inline C *
_ptr_t<HPtr, Ptr, C>::get_context()
{
    return ctx_;
}

template <typename HPtr, typename Ptr, typename C>
inline const C *
_ptr_t<HPtr, Ptr, C>::get_context() const
{
    return ctx_;
}

template <typename HPtr, typename Ptr, typename C>
inline bool
_ptr_t<HPtr, Ptr, C>::is_host_ptr() const
{
    return ctx_ == NULL;
}

template <typename HPtr, typename Ptr, typename C>
inline bool
_ptr_t<HPtr, Ptr, C>::is_device_ptr() const
{
    return ctx_ != NULL;
}

}}

#endif /* PTR_IMPL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
