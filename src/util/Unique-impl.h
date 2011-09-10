#ifndef GMAC_UTIL_UNIQUE_IMPL_H_
#define GMAC_UTIL_UNIQUE_IMPL_H_

#include <cstdio>

namespace __impl { namespace util {

#ifdef DEBUG
inline
Unique::Unique()
{
    id_ = unsigned(AtomicInc(Count_)) - 1;
    printf("Created Mode with id %u\n", id_);
}

inline
unsigned
Unique::getId() const
{
    return id_;
}
#endif

}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
