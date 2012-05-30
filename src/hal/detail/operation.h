#ifndef GMAC_HAL_TYPES_OPERATION_H_
#define GMAC_HAL_TYPES_OPERATION_H_

#include "hal/error.h"

#include "util/lock.h"
#include "util/trigger.h"
#include "util/smart_ptr.h"
#include "util/unique.h"

namespace __impl { namespace hal {
    
namespace detail {

class stream;
class GMAC_LOCAL operation {
public:
    enum type {
        TransferToHost,
        TransferToDevice,
        TransferHost,
        TransferDevice,
        Kernel,
        Invalid
    };

    enum state {
        Queued,
        Submit,
        Start,
        End,
        None
    };

protected:
#ifdef USE_TRACE
    hal::time_t timeQueued_;
    hal::time_t timeSubmit_;
    hal::time_t timeStart_;
    hal::time_t timeEnd_;
#endif

    type type_;
    bool async_;
    bool synced_;
    state state_;
    hal::error err_;

    operation(type t, bool async) :
        type_(t),
        async_(async),
        state_(None)
    {
    }

public:
    virtual ~operation()
    {}

    type get_type() const
    {
        return type_;
    }

    virtual hal::error sync() = 0;
    virtual state get_state() = 0;

#ifdef USE_TRACE
    hal::time_t get_time_queued() const;
    hal::time_t get_time_submit() const;
    hal::time_t get_time_start() const;
    hal::time_t get_time_end() const;
#endif

#if 0
    virtual void set_barrier(stream &s) = 0;
#endif

    bool is_async() const
    {
        return async_;
    }

    virtual bool is_host() const
    {
        return false;
    }
};

}}}

#endif /* EVENT_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
