#include "processing_unit.h"
#include "hal/detail/types.h"

namespace __impl { namespace hal { namespace detail { namespace phys {

processing_unit::processing_unit(platform &platform, type t, aspace &as, virt::scheduler &sched) :
    platform_(platform),
    type_(t),
    as_(as),
    sched_(sched)
{
    TRACE(LOCAL, FMT_ID2" Create", get_print_id2());
    as.add_processing_unit(*this);
}

void
processing_unit::add_memory_connection(const memory_connection &connection)
{
    connections_.insert(connection);
    if (connection.direct) {
        connection.mem->add_attached_unit(*this);
    }
}

memory &
processing_unit::get_preferred_memory()
{
    memory_connection ret = *connections_.begin();
    for (auto &c : connections_) {
        if (!ret.direct && c.direct) {
            ret = c;
        } else if (c.direct) {
            if (c.latency < ret.latency) {
                ret = c;
            }
        }
    }

    return *ret.mem;
}

const memory &
processing_unit::get_preferred_memory() const
{
    memory_connection ret = *connections_.begin();
    for (auto &c : connections_) {
        if (!ret.direct && c.direct) {
            ret = c;
        } else if (c.direct) {
            if (c.latency < ret.latency) {
                ret = c;
            }
        }
    }

    return *ret.mem;
}



}}}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
