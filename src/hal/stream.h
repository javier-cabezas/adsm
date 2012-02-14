#ifndef GMAC_HAL_TYPES_STREAM_H_
#define GMAC_HAL_TYPES_STREAM_H_

#include <algorithm>
#include <queue>

#include "trace/logger.h"
#include "util/gmac_base.h"
#include "util/lock.h"

#include "event.h"

namespace __impl { namespace hal {

namespace detail {

class GMAC_LOCAL stream :
    public util::gmac_base<stream>,
    public gmac::util::spinlock<stream> {

protected:
    aspace &aspace_;

    event_ptr lastEvent_;

    stream(aspace &context);

public:
    enum state {
        Empty,
        Running
    };

    aspace &get_aspace();

    event_ptr get_last_event();
    void set_last_event(event_ptr event);

    virtual state query() = 0;
    virtual gmacError_t sync() = 0;
};

}

}}

#endif /* STREAM_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
