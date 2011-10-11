#ifndef GMAC_HAL_TYPES_DETAIL_H_
#define GMAC_HAL_TYPES_DETAIL_H_

#include "config/common.h"

#include "include/gmac/types.h"

namespace __impl { namespace hal {

typedef uint64_t time_t;
time_t get_timestamp();

namespace detail {

template <typename C, typename S, typename E>
struct backend_traits
{
    typedef C context;
    typedef S stream;
    typedef E event;
};

template <typename D, typename B>
class aspace_t {
    typename B::context context_;
    D &device_;

protected:
    aspace_t(typename B::context context, D &device);

public:
    D &get_device();
    typename B::context &operator()();
    const typename B::context &operator()() const;
};

template <typename D, typename B>
class stream_t {
    typedef aspace_t<D, B> aspace_parent_t;
    typename B::stream stream_;
    aspace_parent_t &aspace_;

protected:
    stream_t(typename B::stream stream, aspace_parent_t &aspace);

public:
    aspace_parent_t &get_address_space();
    typename B::stream &operator()();
    const typename B::stream &operator()() const;
};

template <typename D, typename B>
class event_t {
    typedef stream_t<D, B> stream_parent_t;

private:
    stream_parent_t &stream_;
    gmacError_t err_;
    bool isAsynchronous_;
    bool synced_;

protected:
    hal::time_t timeQueued_;
    hal::time_t timeSubmit_;
    hal::time_t timeStart_;
    hal::time_t timeEnd_;

    event_t(stream_parent_t &stream, gmacError_t err = gmacSuccess);

    void set_error(gmacError_t ret);

public:
    stream_parent_t &get_stream();

    gmacError_t getError() const;
    hal::time_t get_time_queued() const;
    hal::time_t get_time_submit() const;
    hal::time_t get_time_start() const;
    hal::time_t get_time_end() const;
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
