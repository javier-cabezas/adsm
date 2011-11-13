#ifndef GMAC_API_OPENCL_HPE_MODULE_IMPL_H_
#define GMAC_API_OPENCL_HPE_MODULE_IMPL_H_

namespace __impl { namespace hal { namespace opencl {

inline const kernel_t *
module::get_kernel(gmac_kernel_id_t key) const
{
    map_kernel::const_iterator k;
    k = kernels_.find(key);
    if(k == kernels_.end()) return NULL;
    return k->second;
}

inline const kernel_t *
module::get_kernel(const std::string &name) const
{
    FATAL("Not implemented");
    return NULL;
}

}}}

#endif
