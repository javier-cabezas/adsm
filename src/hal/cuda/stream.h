#ifndef GMAC_HAL_CUDA_STREAM_H_
#define GMAC_HAL_CUDA_STREAM_H_

#include <cuda.h>
#include <driver_types.h>
#include <vector_types.h>

#include "util/unique.h"

namespace __impl { namespace hal {
    
namespace cuda {

typedef hal::detail::event hal_event;

class GMAC_LOCAL stream :
    public hal::detail::stream {

    typedef hal::detail::stream parent;

    CUstream stream_;

public:
    stream(virt::aspace &as, CUstream stream);

    virt::aspace &get_aspace();

    parent::state query();
    hal::error sync();

    CUstream operator()();
    const CUstream operator()() const;

    hal::error set_barrier(hal_event &e)
    {
        hal::error ret = hal::error::HAL_SUCCESS;
        hal::detail::operation *_op = e.get_last_operation();
        if (_op->is_host() == false) {
            operation *op = reinterpret_cast<operation *>(_op);
            op->set_barrier(*this);
        } else {
            ret = _op->sync();
        }
    
        return ret;
    }

    hal::error set_barrier(list_event_detail &events)
    {
        hal::error ret = hal::error::HAL_SUCCESS;
        for (auto e : events) {
            ret = set_barrier(*e);
        }

        return ret;
    }
};

}}}

#endif /* GMAC_HAL_CUDA_STREAM_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
