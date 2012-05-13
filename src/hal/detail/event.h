#ifndef GMAC_HAL_TYPES_EVENT_H_
#define GMAC_HAL_TYPES_EVENT_H_

#include "hal/error.h"

#include "util/trigger.h"
#include "util/smart_ptr.h"

namespace __impl { namespace hal { namespace detail {

namespace virt {
    class aspace;
}

class stream;
class _event;
typedef util::shared_ptr<_event> event_ptr;

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

    virtual void set_barrier(stream &s) = 0;
};

class GMAC_LOCAL _event :
    public util::list_trigger<>,
    public util::unique<_event>,
    public gmac::util::lock_rw<_event > {

public:
    enum type {
        Memory,
        Kernel,
        IO,
        Invalid
    };

    typedef operation::state state;

protected:
    bool async_;
    bool synced_;
    type type_;
    state state_;
    hal::error err_;

#ifdef USE_TRACE
    hal::time_t timeBase_;
#endif
    _event(bool async, type t);

public:
    typedef std::list<operation *> list_operation;
    list_operation operations_;
    list_operation::iterator syncOpBegin_;

    type get_type() const;
    virtual state get_state() = 0;

#if 0
    hal::time_t get_time_queued() const;
    hal::time_t get_time_submit() const;
    hal::time_t get_time_start() const;
    hal::time_t get_time_end() const;
#endif

    bool is_synced() const;

    virtual hal::error sync() = 0;

    virtual void set_synced() = 0;

    operation *get_last_operation()
    {
        if (operations_.size() == 0) {
            return NULL;
        } else {
            return *std::prev(operations_.end());
        }
    }
};

class GMAC_LOCAL list_event :
    protected std::list<event_ptr> {
    typedef std::list<event_ptr> parent;
public:
    typedef parent::const_iterator const_iterator;

    void add_event(event_ptr event)
    {
        parent::push_back(event);
    }

    const_iterator begin() const
    {
        return parent::begin();
    }

    const_iterator end() const
    {
        return parent::end();
    }

    virtual hal::error sync() = 0;

    size_t size() const;
};

}}}

#endif /* EVENT_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
