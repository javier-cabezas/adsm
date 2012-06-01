#ifndef GMAC_HAL_CUDA_VIRT_CONTEXT_IMPL_H_
#define GMAC_HAL_CUDA_VIRT_CONTEXT_IMPL_H_

namespace __impl { namespace hal { namespace cuda { namespace virt {

inline
context::context(virt::aspace *as, hal_priority prio, hal::error &err) :
    hal_context(as, prio)
{
    if (as != nullptr) {
        as->set();

        CUstream s;
        CUresult res = cuStreamCreate(&s, 0);
        if (res != CUDA_SUCCESS) {
            err = error_to_hal(res);
            return;
        }

        stream_ = new stream(*as, s);
    }
}

inline
context::~context()
{
    if (aspace_ != nullptr) {
        delete stream_;
    }
}

inline
hal_event_ptr
context::queue(const hal_kernel &_k, hal_kernel::config &_config, hal_kernel::arg_list &_args, hal::error &err)
{
    const code::kernel &k = reinterpret_cast<const code::kernel &>(_k);
    code::kernel::config &config = reinterpret_cast<code::kernel::config &>(_config);
    code::kernel::arg_list &args = reinterpret_cast<code::kernel::arg_list &>(_args);

    code::kernel::launch *launch = k.launch_config(config, args, *stream_, err);
    hal_event_ptr ret = launch->execute(err);

    return ret;
}

}}}}

#endif /* CONTEXT */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
