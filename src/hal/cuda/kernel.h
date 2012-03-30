#ifndef GMAC_HAL_CUDA_KERNEL_H_
#define GMAC_HAL_CUDA_KERNEL_H_

#include <cuda.h>
#include <driver_types.h>
#include <vector_types.h>

#include "util/ReusableObject.h"
#include "util/unique.h"

#include "types.h"

namespace __impl { namespace hal {
    
namespace cuda { namespace code {

typedef hal::detail::event_ptr hal_event_ptr;
typedef hal::detail::code::kernel hal_kernel;

class GMAC_LOCAL kernel :
    public hal_kernel {

    typedef hal_kernel parent;
    
public:
    class launch;

    class GMAC_LOCAL config :
        public hal_kernel::config {
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
        public hal_kernel::arg_list {
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
        public parent::launch {

        config &config_;
        arg_list &args_;
        stream &stream_;

    public:
        launch(kernel &parent, config &conf, arg_list &args, stream &s);

        stream &get_stream();
        const stream &get_stream() const;

        const kernel &get_kernel() const;

        const config &get_config() const;
        const arg_list &get_arg_list() const;

        hal_event_ptr execute(list_event_detail &dependencies, gmacError_t &err);
        hal_event_ptr execute(hal_event_ptr event, gmacError_t &err);
        hal_event_ptr execute(gmacError_t &err);
    };

    CUfunction kernel_;

    kernel(CUfunction func, const std::string &name);

    CUfunction &operator()();
    const CUfunction &operator()() const;
    //launch &launch_config(Parent::config &config, Parent::arg_list &args, stream &stream);
};

}}}}

#endif /* GMAC_HAL_CUDA_KERNEL_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
