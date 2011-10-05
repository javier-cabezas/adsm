#ifndef GMAC_UTIL_UNIQUE_IMPL_H_
#define GMAC_UTIL_UNIQUE_IMPL_H_

#include <cstdio>

namespace __impl { namespace util {

template <typename T, typename R>
Atomic Unique<T, R>::Count_ = 0;

#ifdef DEBUG
template <typename T, typename R>
Atomic UniqueDebug<T, R>::Count_ = 0;
#endif

template <typename T, typename R>
inline
Unique<T, R>::Unique() :
    id_(R(AtomicInc(Count_) - 1))
{
}

template <typename T, typename R>
inline
R
Unique<T, R>::getId() const
{
    return id_;
}

#ifdef DEBUG
template <typename T, typename R>
inline
UniqueDebug<T, R>::UniqueDebug() :
    id_(R(AtomicInc(Count_) - 1))
{
}

template <typename T, typename R>
inline
R
UniqueDebug<T, R>::getDebugId() const
{
    return id_;
}
#endif



}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
