#ifndef GMAC_HAL_DETAIL_VIRT_CONTEXT_H_
#define GMAC_HAL_DETAIL_VIRT_CONTEXT_H_

#include "trace/logger.h"
#include "util/gmac_base.h"
#include "util/lock.h"

namespace __impl { namespace hal { namespace detail {

namespace virt {

class aspace;

typedef unsigned priority;

class GMAC_LOCAL context :
    public util::gmac_base<context>,
    public gmac::util::spinlock<context> {

    typedef gmac::util::spinlock<context> lock;

protected:
    aspace *aspace_;
    priority prio_;

    context(aspace *as, priority prio);

public:
    enum state {
        Empty,
        Wait,
        Running,
    };

    virtual ~context()
    {
    }

    aspace *get_aspace();
    const aspace *get_aspace() const;

    void set_priority(priority prio);
    priority get_priority() const;

    virtual hal::error migrate(virt::aspace &as) = 0;

    virtual state get_state() = 0;
    virtual event_ptr queue(const code::kernel &k, code::kernel::config &config, code::kernel::arg_list &args, hal::error &err) = 0;
};

}}}}

#endif /* STREAM_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
