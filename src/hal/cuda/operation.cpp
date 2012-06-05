#include "types.h"

namespace __impl { namespace hal { namespace cuda {

operation::state
operation::get_state()
{
    as_.set();

    CUresult res = cuEventQuery(eventEnd_);

    // Advance untile we find a not ready event
    if (res == CUDA_ERROR_NOT_READY) {
        return End;
    } else {
        return Start;
    }
}

void
operation::reset(type t, bool async, stream &s)
{
    async_ = async;
    type_ = t;
    err_ = hal::error::HAL_SUCCESS;
    synced_ = false;
    stream_ = &s;
}

hal::error
list_event::sync()
{
    hal::error ret = hal::error::HAL_SUCCESS;
    for (parent::const_iterator it  = parent::begin();
                                it != parent::end();
                              ++it) {
        ret = (*it)->sync();
        if (ret != hal::error::HAL_SUCCESS) break;
    }

    return ret;
}

#if 0
void
list_event::set_barrier(hal_stream &_stream)
{
    stream &s = reinterpret_cast<stream &>(_stream);
    s.get_aspace().set();

    for (parent::iterator it  = parent::begin();
                          it != parent::end();
                        ++it) {
        (**it).set_synced();
    }
}
#endif

void
list_event::set_synced()
{
    for (parent::const_iterator it  = parent::begin();
                                it != parent::end();
                              ++it) {
        (**it).set_synced();
    }
}


}}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
