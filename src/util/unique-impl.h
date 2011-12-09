#ifndef GMAC_UTIL_UNIQUE_IMPL_H_
#define GMAC_UTIL_UNIQUE_IMPL_H_

#include <cstdio>

namespace __impl { namespace util {

template <typename T, typename R>
Atomic unique<T, R>::Count_ = 0;

inline
default_id::default_id(unsigned long val) :
    val_(val)
{
}

inline
unsigned long
default_id::print() const
{
    return val_;
}

template <typename T, typename R>
inline
unique<T, R>::unique() :
    id_(R(AtomicInc(Count_) - 1))
{
}

template <typename T, typename R>
inline
R
unique<T, R>::get_id() const
{
    return id_;
}

template <typename T, typename R>
inline
unsigned long
unique<T, R>::get_print_id() const
{
    return id_.print();
}

}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
