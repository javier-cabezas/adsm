#ifndef GMAC_HAL_CUDA_STREAM_H_
#define GMAC_HAL_CUDA_STREAM_H_

#include <cuda.h>
#include <driver_types.h>
#include <vector_types.h>

#include "util/unique.h"

namespace __impl { namespace hal {
    
namespace cuda {

typedef hal::detail::_event hal_event;

class GMAC_LOCAL stream :
    public hal::detail::stream {

    typedef hal::detail::stream parent;

    CUstream stream_;

public:
    stream(virt::aspace &as, CUstream stream);

    virt::aspace &get_aspace();

    parent::state query();
    gmacError_t sync();

    CUstream operator()();
    const CUstream operator()() const;

    gmacError_t set_barrier(hal_event &event)
    {
        hal::detail::operation *op = event.get_last_operation();
        op->set_barrier(*this);
    
        return gmacSuccess;
    }

    gmacError_t set_barrier(list_event_detail &events)
    {
        for (list_event::const_iterator it  = events.begin();
                                        it != events.end();
                                      ++it) {
#if 0
            (**it).set_barrier(get_aspace(), stream_);
#endif
            hal::detail::operation *op = (*it)->get_last_operation();
            op->set_barrier(*this);
            
        }
        return gmacSuccess;
    }
};

}}}

#endif /* GMAC_HAL_CUDA_STREAM_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
