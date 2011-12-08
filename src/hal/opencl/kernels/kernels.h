#ifndef GMAC_HAL_OPENCL_KERNELS_KERNELS_H_
#define GMAC_HAL_OPENCL_KERNELS_KERNELS_H_

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

static
const char *memset_kernel = "                                               \
    __kernel                                                                \
    void gmac_memset_internal__(__global char4 *mem, unsigned off, int val) \
    {                                                                       \
        unsigned id = get_global_id(0);                                     \
                                                                            \
        char4 set = (char4)((char) val);                                    \
                                                                            \
        mem[off + id] = set;                                                \
    }";

static
const char *KernelsGmac_[] = {
    memset_kernel
};

#define gmac_memset "gmac_memset_internal__"

namespace __impl { namespace hal { namespace opencl {

}}}

#endif /* GMAC_HAL_OPENCL_KERNELS_KERNELS_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
