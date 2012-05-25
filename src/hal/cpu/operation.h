#ifndef GMAC_HAL_CPU_OPERATION_H_
#define GMAC_HAL_CPU_OPERATION_H_

namespace __impl { namespace hal { namespace cpu {

typedef hal::detail::list_event list_event_detail;

class GMAC_LOCAL operation :
    public hal::detail::operation
{
    typedef hal::detail::operation parent;
public:
    operation(parent::type t, bool async) :
        parent(t, async)
    {}

    template <typename Func, typename... Args>
    auto execute(Func f, Args... args) -> decltype(f(args...));
    
    hal::error sync();
    parent::state get_state();
    void set_barrier(hal::detail::stream &);
};

}}}

#endif /* EVENT_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
