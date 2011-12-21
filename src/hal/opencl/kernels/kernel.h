#ifndef GMAC_HAL_OPENCL_KERNEL_H_
#define GMAC_HAL_OPENCL_KERNEL_H_

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "hal/types-detail.h"

#include "util/unique.h"

namespace __impl { namespace hal { namespace opencl {

class GMAC_LOCAL kernel_t :
    public hal::detail::kernel_t<backend_traits, implementation_traits> {

    typedef hal::detail::kernel_t<backend_traits, implementation_traits> Parent;
    
public:
    class launch;

    class GMAC_LOCAL config :
        public hal::detail::kernel_t<backend_traits, implementation_traits>::config {
        friend class launch;

        const size_t *dimsOffset_;
        const size_t *dimsGlobal_;
        const size_t *dimsGroup_;
    public:
        config() {}
        config(unsigned ndims, const size_t *offset, const size_t *global, const size_t *group);

        const size_t *get_dims_offset() const;
        const size_t *get_dims_global() const;
        const size_t *get_dims_group() const;

    };

    class GMAC_LOCAL arg_list :
        public hal::detail::kernel_t<backend_traits, implementation_traits>::arg_list {
        friend class launch;

        unsigned nArgs_;
        //const void *params_[256];

        typedef std::pair<cl_mem, unsigned> cl_mem_ref;
        typedef std::map<host_ptr, cl_mem_ref> map_subbuffer;
        class map_global_subbuffer :
            protected std::map<cl_context, map_subbuffer>,
            protected gmac::util::spinlock<map_global_subbuffer> {
        public:
            typedef std::map<cl_context, map_subbuffer> Parent;
            typedef Parent::iterator iterator;
            map_global_subbuffer() :
                gmac::util::spinlock<map_global_subbuffer>("map_global_subbuffer")
            {
            }

            iterator
            find_context(cl_context context)
            {  
                lock();
                Parent::iterator itGlobalMap = this->find(context);
                if (itGlobalMap == this->end()) {
                    this->insert(Parent::value_type(context, map_subbuffer()));
                    itGlobalMap = this->find(context);
                }
                unlock();
                return itGlobalMap;
            }
        };

        typedef std::pair<cl_context, map_subbuffer::iterator> cache_entry;
        typedef std::map<host_ptr, cache_entry> cache_subbuffer;

        static map_global_subbuffer mapSubBuffer_;

        cache_subbuffer cacheSubBuffer_;

    public:
        arg_list() :
            nArgs_(0)
        {
        }

        unsigned get_nargs() const;
        gmacError_t set_arg(kernel_t &k, const void *arg, size_t size, unsigned index);
        cl_mem get_subbuffer(cl_context context, host_ptr host, ptr_t dev, size_t size);
    };

    class GMAC_LOCAL launch :
        public hal::detail::kernel_t<backend_traits, implementation_traits>::launch {
 
        event_t execute(unsigned nevents, const cl_event *events, gmacError_t &err);

        unsigned nArgs_;
        const void *params_[256];
    public:
        launch(kernel_t &parent, config &conf, arg_list &args, stream_t &stream);

        event_t execute(list_event_detail &dependencies, gmacError_t &err);
        event_t execute(event_t event, gmacError_t &err);
        event_t execute(gmacError_t &err);
    };

    kernel_t(cl_kernel func, const std::string &name);

    launch &launch_config(Parent::config &config, Parent::arg_list &args, stream_t &stream);
};

}}}

#endif /* GMAC_HAL_OPENCL_KERNEL_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
