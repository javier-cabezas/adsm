#ifndef GMAC_HAL_TYPES_STREAM_H_
#define GMAC_HAL_TYPES_STREAM_H_

#include <algorithm>
#include <queue>

#include "trace/logger.h"
#include "util/gmac_base.h"
#include "util/lock.h"

namespace __impl { namespace hal {

namespace detail {

template <typename I>
class GMAC_LOCAL stream_t :
    public util::gmac_base<stream_t<I> >,
    public gmac::util::spinlock<stream_t<I> > {

    typedef typename I::context context_parent_t;
    friend class I::context;
protected:
    context_parent_t &context_;

    typename I::event_ptr lastEvent_;

    stream_t(context_parent_t &context);

public:
    enum state {
        Empty,
        Running
    };

    context_parent_t &get_context();

    typename I::event_ptr get_last_event();
    void set_last_event(typename I::event_ptr event);

    virtual state query() = 0;
    virtual gmacError_t sync() = 0;
};

}

}}

#endif /* STREAM_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
