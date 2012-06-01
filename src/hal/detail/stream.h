#ifndef GMAC_HAL_DETAIL_STREAM_H_
#define GMAC_HAL_DETAIL_STREAM_H_

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
    virt::aspace &aspace_;

    stream(virt::aspace &as);

public:
    enum state {
        Empty,
        Running
    };

    virt::aspace &get_aspace();

    virtual state query() = 0;
    virtual hal::error sync() = 0;
};

}

}}

#endif /* STREAM_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
