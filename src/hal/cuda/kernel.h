#ifndef GMAC_HAL_CUDA_KERNEL_H_
#define GMAC_HAL_CUDA_KERNEL_H_

#include <cuda.h>
#include <driver_types.h>
#include <vector_types.h>

#include "hal/types-detail.h"

#include "util/unique.h"

namespace __impl { namespace hal {
    
namespace cuda {

class GMAC_LOCAL kernel_t :
    public hal::detail::kernel_t<backend_traits, implementation_traits> {

    typedef hal::detail::kernel_t<backend_traits, implementation_traits> Parent;
    
public:
    class launch;

    class GMAC_LOCAL config :
        public hal::detail::kernel_t<backend_traits, implementation_traits>::config {
        friend class launch;

        dim3 dimsGlobal_;
        dim3 dimsGroup_;
        size_t memShared_;
    public:
        config() {}
        config(dim3 global, dim3 group, size_t shared, cudaStream_t tokens);

        const dim3 &get_dims_global() const;
        const dim3 &get_dims_group() const;
    };

    class GMAC_LOCAL arg_list :
        public hal::detail::kernel_t<backend_traits, implementation_traits>::arg_list {
        friend class launch;

        unsigned nArgs_;
        const void *params_[256];

    protected:
        arg_list() :
            nArgs_(0)
        {
        }

    public:
        unsigned get_nargs() const;
        gmacError_t push_arg(const void *arg, size_t size);
    };

    class GMAC_LOCAL launch :
        public hal::detail::kernel_t<backend_traits, implementation_traits>::launch {

    public:
        launch(kernel_t &parent, config &conf, arg_list &args, stream_t &stream);

        event_t execute(list_event_detail &dependencies, gmacError_t &err);
        event_t execute(event_t event, gmacError_t &err);
        event_t execute(gmacError_t &err);
    };

    kernel_t(CUfunction func, const std::string &name);

    launch &launch_config(Parent::config &config, Parent::arg_list &args, stream_t &stream);
};

}}}

#endif /* GMAC_HAL_CUDA_KERNEL_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
