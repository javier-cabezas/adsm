#ifndef GMAC_HAL_TYPES_DETAIL_H_
#define GMAC_HAL_TYPES_DETAIL_H_

#include <string>

#include "config/common.h"

#include "include/gmac/types.h"

#ifndef _MSC_VER
#include <sys/time.h>
#endif

namespace __impl { namespace hal {

typedef uint64_t time_t;

#ifdef _MSC_VER
static inline
time_t get_timestamp()
{
    FILETIME ft;
    GetSystemTimeAsFileTime(&ft);

    unsigned __int64 tmp = 0;
    tmp |= ft.dwHighDateTime;
    tmp <<= 32;
    tmp |= ft.dwLowDateTime;
    tmp -= DELTA_EPOCH_IN_MICROSECS;
    tmp /= 10;

    return time_t(tmp);
}
#else
static inline
time_t get_timestamp()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    time_t ret;

    ret = tv.tv_sec * 1000000 + tv.tv_usec;
    return ret;
}
#endif

class GMAC_LOCAL device_input {
public:
    virtual bool read(void *ptr, size_t count) = 0;
};

class GMAC_LOCAL device_output {
public:
    virtual bool write(void *ptr, size_t count) = 0;
};

namespace detail {

template <typename C, typename S, typename E, typename K, typename KC>
struct GMAC_LOCAL backend_traits
{
    typedef C context;
    typedef S stream;
    typedef E event;
    typedef K kernel;
    typedef KC kernel_config;
};

template <typename CD, typename C, typename S, typename K, typename T, typename V, typename R, typename E, typename EP, typename EL, typename B, typename P> 
struct GMAC_LOCAL implementation_traits
{
    typedef CD coherence_domain;
    typedef C context;
    typedef S stream;
    typedef K kernel;
    typedef T texture;
    typedef V variable;
    typedef R code_repository;
    typedef E event;
    typedef EP event_ptr;
    typedef EL event_list;
    typedef B buffer;
    typedef P ptr;
};

}

}}

#include "stream.h"
#include "context.h"
#include "kernel.h"
#include "event.h"

#include "context-impl.h"
#include "stream-impl.h"
#include "kernel-impl.h"
#include "event-impl.h"

#endif /* TYPES_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
