#ifndef GMAC_HAL_CUDA_STREAM_H_
#define GMAC_HAL_CUDA_STREAM_H_

#include <cuda.h>
#include <driver_types.h>
#include <vector_types.h>

#include "util/unique.h"

namespace __impl { namespace hal {
    
namespace cuda {

class GMAC_LOCAL stream :
    public hal::detail::stream {

    typedef hal::detail::stream parent;

    bool valid_;
    CUstream stream_;

public:
    stream(virt::aspace &as);
    stream(CUstream stream, virt::aspace &as);

    void set_cuda_stream(CUstream s);

    virt::aspace &get_aspace();

    parent::state query();
    gmacError_t sync();

    CUstream &operator()();
    const CUstream &operator()() const;
};

}}}

#include "module.h"

#endif /* GMAC_HAL_CUDA_STREAM_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
