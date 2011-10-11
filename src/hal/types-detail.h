#ifndef GMAC_HAL_TYPES_DETAIL_H_
#define GMAC_HAL_TYPES_DETAIL_H_

#include "config/common.h"

#include "include/gmac/types.h"

namespace __impl { namespace hal {

namespace detail {

template <typename C, typename S, typename E>
struct backend_traits
{
    typedef C Context;
    typedef S Stream;
    typedef E Event;
};

template <typename D, typename B>
class aspace_t {
    typename B::Context context_;
    D &device_;

protected:
    aspace_t(typename B::Context context, D &device);
public:
    D &getDevice();
    typename B::Context &operator()();
    const typename B::Context &operator()() const;
};

template <typename D, typename B>
class stream_t {
    typedef aspace_t<D, B> aspace_parent_t;
    typename B::Stream stream_;
    aspace_parent_t &aspace_;

protected:
    stream_t(typename B::Stream stream, aspace_parent_t &aspace);

public:
    aspace_parent_t &getPASpace();
    typename B::Stream &operator()();
    const typename B::Stream &operator()() const;
};

template <typename D, typename B>
class event_t {
    typedef stream_t<D, B> stream_parent_t;

public:
    typedef long_t time_t;

private:
    stream_parent_t &stream_;
    gmacError_t err_;
    bool isAsynchronous_;
    bool synced_;

protected:
    time_t start_;
    time_t end_;

    event_t(stream_parent_t &stream, gmacError_t err = gmacSuccess);

    void setError(gmacError_t ret);

public:
    stream_parent_t &getStream();

    gmacError_t getError() const;
    time_t getStartTime() const;
    time_t getEndTime() const;
    time_t getElapsedTime() const;
};

template <typename D, typename B>
class async_event_t :
    public event_t<D, B> {
    typedef stream_t<D, B> stream_parent_t;

private:
    bool synced_;

protected:
    async_event_t(stream_parent_t &stream, gmacError_t err = gmacSuccess);

public:
    virtual gmacError_t sync() = 0;
    bool isSynced() const;
    void setSynced(bool synced);
};

}
}}

#include "types-impl.h"

#endif /* TYPES_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
