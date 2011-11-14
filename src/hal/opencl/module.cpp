#include "hpe/init.h"

#include "module.h"
#include "device.h"

namespace __impl { namespace hal {
        
extern opencl::map_platform_repository Modules_;

namespace opencl {

module_descriptor::vector_module_descriptor module_descriptor::ModuleDescriptors_;

module_descriptor::module_descriptor(const std::string &code,
                                     const std::string &flags) :
    code_(code),
    flags_(flags)
{
    TRACE(LOCAL, "Creating module descriptor");
    ModuleDescriptors_.push_back(this);
}

const std::string &
module_descriptor::get_code() const
{
    return code_;
}

const std::string &
module_descriptor::get_compilation_flags() const
{
    return flags_;
}

code_repository
module_descriptor::create_modules(platform &plat, gmacError_t &err)
{
    TRACE(GLOBAL, "Creating modules");

    err = gmacSuccess;
    code_repository repositories;

    vector_module_descriptor::const_iterator it;
    for (it = ModuleDescriptors_.begin(); it != ModuleDescriptors_.end(); it++) {
        module *ptr = new module(**it, plat, err);
        if (err != gmacSuccess) break;
        repositories.push_back(ptr);
    }

#if 0
    vector_module_descriptor::const_iterator it;
    for (it = ModuleDescriptors_.begin(); it != ModuleDescriptors_.end(); it++) {
        TRACE(GLOBAL, "Creating module: %p", (*it)->fatBin_);
        modules.push_back(new cuda::module(*(*it)));
    }
#endif
    return repositories;
}

static const int CUDA_MAGIC = 0x466243b1;

struct GMAC_LOCAL FatBinDesc {
    int magic; int v; const unsigned long long* data; char* f;
};

module::module(const module_descriptor &descriptor, platform &plat, gmacError_t &err)
{
    cl_int res;

    const std::string &code = descriptor.get_code();
    const char *stringCode = code.c_str();
    size_t size = code.size();

    cl_program program = clCreateProgramWithSource(plat.get_context(), 1, &stringCode, &size, &res);
    if (res != CL_SUCCESS) {
        err = error(res);
        return;
    }

    cl_device_id *deviceIds = plat.get_cl_device_array();
    size_t ndevices = plat.get_ndevices();

    res = clBuildProgram(program, ndevices, deviceIds, descriptor.get_compilation_flags().c_str(), NULL, NULL);

    if (res == CL_SUCCESS) {
        cl_uint nkernels;
        res = clCreateKernelsInProgram(program, 0, NULL, &nkernels);

        if (res == CL_SUCCESS) {
            cl_kernel *kernels = new cl_kernel[nkernels];
            res = clCreateKernelsInProgram(program, nkernels, kernels, NULL);
            ASSERTION(res == CL_SUCCESS);

            for (cl_uint i = 0; i < nkernels; i++) {
                size_t size;
                res = clGetKernelInfo(kernels[i], CL_KERNEL_FUNCTION_NAME, 0, NULL, &size);
                ASSERTION(res == CL_SUCCESS);
                char *name = new char[size + 1];
                res = clGetKernelInfo(kernels[i], CL_KERNEL_FUNCTION_NAME, size, name, NULL);
                ASSERTION(res == CL_SUCCESS);
                name[size] = '\0';

                printf("Registering kernel %s\n", name);
                kernels_.insert(map_kernel::value_type(name, new kernel_t(kernels[i], std::string(name))));
            }
        }
    }

    err = error(res);
    delete []deviceIds;
}

module::~module()
{
    map_kernel::iterator it;
    for (it = kernels_.begin(); it != kernels_.end(); it++) {
        delete it->second;
    }
#ifdef CALL_CUDA_ON_DESTRUCTION
    std::vector<CUmodule>::const_iterator m;
    for(m = mods_.begin(); m != mods_.end(); m++) {
        CUresult ret = cuModuleUnload(*m);
        ASSERTION(ret == CUDA_SUCCESS);
    }
    mods_.clear();
#endif

    // TODO: remove objects from maps
#if 0
    variables_.clear();
    constants_.clear();
    textures_.clear();
#endif
}

}}}
