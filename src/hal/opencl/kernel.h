#ifndef GMAC_HAL_OPENCL_KERNEL_H_
#define GMAC_HAL_OPENCL_KERNEL_H_

#include <CL/cl.h>

#include "hal/types-detail.h"

#include "util/unique.h"

namespace __impl { namespace hal { namespace opencl {

class GMAC_LOCAL kernel_t :
    public hal::detail::kernel_t<device, backend_traits, implementation_traits> {

    typedef hal::detail::kernel_t<device, backend_traits, implementation_traits> Parent;
    
public:
    class launch;

    class GMAC_LOCAL config :
        public hal::detail::kernel_t<device, backend_traits, implementation_traits>::config {
        friend class launch;

        unsigned nArgs_;

        const size_t *dimsGlobal_;
        const size_t *dimsGroup_;

        const void *params_[256];
    public:
        config(unsigned ndims, const size_t *global, const size_t *group);

        unsigned get_nargs() const;
        const size_t *get_dims_global() const;
        const size_t *get_dims_group() const;

        gmacError_t set_arg(const void *arg, size_t size, unsigned index);
        gmacError_t register_kernel();
    };

    class GMAC_LOCAL launch :
        public hal::detail::kernel_t<device, backend_traits, implementation_traits>::launch {

        event_t execute(unsigned nevents, const cl_event *events, gmacError_t &err);
    public:
        launch(kernel_t &parent, Parent::config &conf, stream_t &stream);

        event_t execute(list_event_detail &dependencies, gmacError_t &err);
        event_t execute(event_t event, gmacError_t &err);
        event_t execute(gmacError_t &err);
    };

    kernel_t(cl_kernel func, const std::string &name);

    launch &launch_config(Parent::config &conf, stream_t &stream);
};

}}}

#endif /* GMAC_HAL_OPENCL_KERNEL_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
