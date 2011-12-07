#ifndef GMAC_API_OPENCL_HPE_MODULE_IMPL_H_
#define GMAC_API_OPENCL_HPE_MODULE_IMPL_H_

namespace __impl { namespace hal { namespace opencl {

inline kernel_t *
module::get_kernel(gmac_kernel_id_t key)
{
    kernel_t *k;
    k = get_kernel(std::string(key));
    return k;
}

inline kernel_t *
module::get_kernel(const std::string &name)
{
    lock();

    kernel_t *t = NULL;
    TRACE(LOCAL, "looking for kernel '%s'...", name.c_str());
    if(std::find(kernels_.begin(), kernels_.end(), name) == kernels_.end()) {
        TRACE(LOCAL, "... not found");
    } else {
        cl_int res;
        map_thread::iterator it = kernelMaps_.find(util::GetThreadId());
        if (it == kernelMaps_.end()) {
            cl_kernel *kernels = new cl_kernel[kernels_.size()];
            res = clCreateKernelsInProgram(program_, kernels_.size(), kernels, NULL);
            ASSERTION(res == CL_SUCCESS);

            map_kernel map;

            for (cl_uint i = 0; i < kernels_.size(); i++) {
                size_t size;
                res = clGetKernelInfo(kernels[i], CL_KERNEL_FUNCTION_NAME, 0, NULL, &size);
                ASSERTION(res == CL_SUCCESS);
                char *name = new char[size + 1];
                res = clGetKernelInfo(kernels[i], CL_KERNEL_FUNCTION_NAME, size, name, NULL);
                ASSERTION(res == CL_SUCCESS);
                name[size] = '\0';

                TRACE(LOCAL, "Registering kernel %s", name);
                t = new kernel_t(kernels[i], std::string(name));
                map.insert(map_kernel::value_type(name, t));
            }

            kernelMaps_.insert(map_thread::value_type(util::GetThreadId(), map));
        } else {
            map_kernel &kernels = it->second;
            map_kernel::iterator it2 = kernels.find(name);

            ASSERTION(it2 != kernels.end(), "Kernel should be registerd for this thread");
            t = it2->second;
        }
    }
    TRACE(LOCAL, "... found!");

    unlock();
    return t;
}

}}}

#endif
