#ifndef GMAC_HAL_TYPES_EVENT_H_
#define GMAC_HAL_TYPES_EVENT_H_

#include "util/trigger.h"
#include "util/smart_ptr.h"

namespace __impl { namespace hal { namespace detail {

namespace virt {
    class aspace;
}

class GMAC_LOCAL _event :
    public util::list_trigger<>,
    public util::unique<_event>,
    public gmac::util::lock_rw<_event > {

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

private:
    virt::aspace &aspace_;

protected:
    bool async_;
    bool synced_;
    type type_;
    state state_;
    gmacError_t err_;

    hal::time_t timeQueued_;
    hal::time_t timeSubmit_;
    hal::time_t timeStart_;
    hal::time_t timeEnd_;

    _event(bool async, type t, virt::aspace &as);

public:
    virt::aspace &get_vaspace();

    type get_type() const;
    virtual state get_state() = 0;

    hal::time_t get_time_queued() const;
    hal::time_t get_time_submit() const;
    hal::time_t get_time_start() const;
    hal::time_t get_time_end() const;

    virtual gmacError_t sync() = 0;
    virtual void set_synced() = 0;

    bool is_synced() const;
};

typedef util::shared_ptr<_event> event_ptr;

class GMAC_LOCAL list_event {
public:
    virtual void add_event(event_ptr event) = 0;

    virtual gmacError_t sync() = 0;
    virtual void set_synced() = 0;
};

}

}}

#endif /* EVENT_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
