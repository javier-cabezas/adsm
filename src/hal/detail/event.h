#ifndef GMAC_HAL_TYPES_EVENT_H_
#define GMAC_HAL_TYPES_EVENT_H_

#include "hal/error.h"

#include "util/lock.h"
#include "util/trigger.h"
#include "util/smart_ptr.h"
#include "util/unique.h"

#include "operation.h"

namespace __impl { namespace hal {
    
namespace cpu {
    class operation;
}

namespace detail {

namespace virt {
    class aspace;
}

class stream;
class event;
typedef util::shared_ptr<event> event_ptr;

class GMAC_LOCAL event :
    public util::list_trigger<>,
    public util::unique<event>,
    public gmac::util::lock_rw<event > {

public:
    enum type {
        Memory,
        Kernel,
        IO,
        Invalid
    };

    typedef operation::state state;

protected:
    bool synced_;
    type type_;
    state state_;
    hal::error err_;

#ifdef USE_TRACE
    hal::time_t timeBase_;
#endif

    typedef std::list<operation *> list_operation;
    list_operation operations_;
    list_operation::iterator syncOpBegin_;

    event(type t);
public:
    ~event()
    {
        for (auto op : operations_) {
            delete op;
        }
    }

    static event_ptr create(type t)
    {
        return event_ptr(new event(t));
    }

    type get_type() const;
    state get_state();

#if 0
    hal::time_t get_time_queued() const;
    hal::time_t get_time_submit() const;
    hal::time_t get_time_start() const;
    hal::time_t get_time_end() const;
#endif

    bool is_synced() const;

    hal::error sync();
    void set_synced();

    operation *get_last_operation()
    {
        if (operations_.size() == 0) {
            return NULL;
        } else {
            return *std::prev(operations_.end());
        }
    }

    template <typename Func, typename Op, typename... Args>
    auto
    queue(const Func &f, Op &op, Args... args) -> decltype(op.execute(f, args...));
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
