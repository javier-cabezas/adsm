#ifndef GMAC_TEST_UNIT_MOCK_HAL_COPY_H_
#define GMAC_TEST_UNIT_MOCK_HAL_COPY_H_

#include <list>

#include "hal/error.h"

namespace __impl { namespace hal {

class GMAC_LOCAL event {
};

typedef std::shared_ptr<event> event_ptr;

typedef std::list<event_ptr> list_event;

static inline
event_ptr
copy(ptr dst, const_ptr src, size_t count, list_event &dependencies, error &err)
{
    return event_ptr(new event);
}

static inline
event_ptr
copy(ptr dst, const_ptr src, size_t count, event_ptr event, error &err)
{
    list_event list({ event });
    return copy(dst, src, count, list, err);
}

static inline
event_ptr
copy(ptr dst, const_ptr src, size_t count, error &err)
{
    list_event list;
    return copy(dst, src, count, list, err);
}

static inline
event_ptr
copy(ptr dst, device_input &input, size_t count, list_event &dependencies, error &err)
{
    return event_ptr(new event);
}

static inline
event_ptr
copy(ptr dst, device_input &input, size_t count, event_ptr event, error &err)
{
    list_event list({ event });
    return copy(dst, input, count, list, err);
}

static inline
event_ptr
copy(ptr dst, device_input &input, size_t count, error &err)
{
    list_event list;
    return copy(dst, input, count, list, err);
}

static inline
event_ptr copy(device_output &output, const_ptr src, size_t count, list_event &dependencies, error &err)
{
    return event_ptr(new event);
}

static inline
event_ptr
copy(device_output &output, const_ptr src, size_t count, event_ptr event, error &err)
{
    list_event list({ event });
    return copy(output, src, count, list, err);
}

static inline
event_ptr
copy(device_output &output, const_ptr src, size_t count, error &err)
{
    list_event list;
    return copy(output, src, count, list, err);
}

static inline
event_ptr
copy_async(ptr dst, const_ptr src, size_t count, list_event &dependencies, error &err)
{
    return event_ptr(new event);
}

static inline
event_ptr
copy_async(ptr dst, const_ptr src, size_t count, event_ptr event, error &err)
{
    list_event list({ event });
    return copy_async(dst, src, count, list, err);
}

static inline
event_ptr
copy_async(ptr dst, const_ptr src, size_t count, error &err)
{
    list_event list;
    return copy_async(dst, src, count, list, err);
}

static inline
event_ptr
copy_async(ptr dst, device_input &input, size_t count, list_event &dependencies, error &err)
{
    return event_ptr(new event);
}

static inline
event_ptr
copy_async(ptr dst, device_input &input, size_t count, event_ptr event, error &err)
{
    list_event list({ event });
    return copy_async(dst, input, count, list, err);
}

static inline
event_ptr
copy_async(ptr dst, device_input &input, size_t count, error &err)
{
    list_event list;
    return copy_async(dst, input, count, list, err);
}

static inline
event_ptr
copy_async(device_output &output, const_ptr src, size_t count, list_event &dependencies, error &err)
{
    return event_ptr(new event);
}

static inline
event_ptr
copy_async(device_output &output, const_ptr src, size_t count, event_ptr event, error &err)
{
    list_event list({ event });
    return copy_async(output, src, count, list, err);
}

static inline
event_ptr
copy_async(device_output &output, const_ptr src, size_t count, error &err)
{
    list_event list;
    return copy_async(output, src, count, list, err);
}

static inline
event_ptr
memset(ptr dst, int c, size_t count, list_event &dependencies, error &err)
{
    return event_ptr(new event);
}

static inline
event_ptr
memset(ptr dst, int c, size_t count, event_ptr event, error &err)
{
    list_event list({ event });
    return memset(dst, c, count, list, err);
}

static inline
event_ptr
memset(ptr dst, int c, size_t count, error &err)
{
    list_event list;
    return memset(dst, c, count, list, err);
}

static inline
event_ptr
memset_async(ptr dst, int c, size_t count, list_event &dependencies, error &err)
{
    return event_ptr(new event);
}

static inline
event_ptr
memset_async(ptr dst, int c, size_t count, event_ptr event, error &err)
{
    list_event list({ event });
    return memset_async(dst, c, count, list, err);
}

static inline
event_ptr
memset_async(ptr dst, int c, size_t count, error &err)
{
    list_event list;
    return memset_async(dst, c, count, list, err);
}

}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
