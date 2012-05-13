#ifndef GMAC_HAL_CPU_EVENT_H_
#define GMAC_HAL_CPU_EVENT_H_

namespace __impl { namespace hal { namespace cpu {

namespace virt {
    class aspace;
}

typedef hal::detail::list_event list_event_detail;

class GMAC_LOCAL _event_t :
    public hal::detail::_event {

    typedef hal::detail::_event parent;

    _event_t(bool async, parent::type t, virt::aspace &as);
public:
    state get_state();

    hal::error sync();
    void set_synced();
};

class GMAC_LOCAL list_event :
    public list_event_detail {
public:
    void add_event(event_ptr event);

    hal::error sync();
    void set_synced();
};

}}}

#endif /* EVENT_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
