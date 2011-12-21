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
    for (it = ModuleDescriptors_.begin(); it != ModuleDescriptors_.end(); ++it) {
        module *ptr = new module(**it, plat, err);
        if (err != gmacSuccess) break;
        repositories.push_back(ptr);
    }

#if 0
    vector_module_descriptor::const_iterator it;
    for (it = ModuleDescriptors_.begin(); it != ModuleDescriptors_.end(); ++it) {
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

module::module(const module_descriptor &descriptor, platform &plat, gmacError_t &err) :
    gmac::util::spinlock<module>("module")
{
    cl_int res;

    const std::string &code = descriptor.get_code();
    const char *stringCode = code.c_str();
    size_t size = code.size();

    program_ = clCreateProgramWithSource(plat.get_context(), 1, &stringCode, &size, &res);
    if (res != CL_SUCCESS) {
        err = error(res);
        return;
    }

    cl_device_id *deviceIds = plat.get_cl_device_array();
    size_t ndevices = plat.get_ndevices();

    res = clBuildProgram(program_, ndevices, deviceIds, descriptor.get_compilation_flags().c_str(), NULL, NULL);

    if (res == CL_SUCCESS) {
        cl_uint nkernels;
        res = clCreateKernelsInProgram(program_, 0, NULL, &nkernels);

        if (res == CL_SUCCESS) {
            cl_kernel *kernels = new cl_kernel[nkernels];
            res = clCreateKernelsInProgram(program_, nkernels, kernels, NULL);
            ASSERTION(res == CL_SUCCESS);

            map_kernel map;

            for (cl_uint i = 0; i < nkernels; ++i) {
                size_t size;
                res = clGetKernelInfo(kernels[i], CL_KERNEL_FUNCTION_NAME, 0, NULL, &size);
                ASSERTION(res == CL_SUCCESS);
                char *name = new char[size + 1];
                res = clGetKernelInfo(kernels[i], CL_KERNEL_FUNCTION_NAME, size, name, NULL);
                ASSERTION(res == CL_SUCCESS);
                name[size] = '\0';

                TRACE(LOCAL, "Registering kernel %s", name);
                map.insert(map_kernel::value_type(name, new kernel_t(kernels[i], std::string(name))));
                kernels_.push_back(name);
            }

            delete []kernels;

            kernelMaps_.insert(map_thread::value_type(util::get_thread_id(), map));
        }
    } else {
        size_t len;
        cl_int tmp = clGetProgramBuildInfo(program_, deviceIds[0],
                CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
        ASSERTION(tmp == CL_SUCCESS);
        char *msg = new char[len + 1];
        tmp = clGetProgramBuildInfo(program_, deviceIds[0],
                CL_PROGRAM_BUILD_LOG, len, msg, NULL);
        ASSERTION(tmp == CL_SUCCESS);
        msg[len] = '\0';
        TRACE(GLOBAL, "Error compiling code accelerator: %d\n%s",
                deviceIds[0], msg);
        delete [] msg;
    }


    err = error(res);
    delete []deviceIds;
}

module::~module()
{
    map_thread::iterator it;

    for (it = kernelMaps_.begin(); it != kernelMaps_.end(); ++it) {
        map_kernel &kernels = it->second;
        map_kernel::iterator it2;

        for (it2 = kernels.begin(); it2 != kernels.end(); ++it2) {
            delete it2->second;
        }
    }
}

kernel_t *
module::get_kernel(gmac_kernel_id_t key)
{
    kernel_t *k;
    k = get_kernel(std::string(key));
    return k;
}

kernel_t *
module::get_kernel(const std::string &name)
{
    lock();

    kernel_t *t = NULL;
    TRACE(LOCAL, "looking for kernel '%s'...", name.c_str());
    if(std::find(kernels_.begin(), kernels_.end(), name) == kernels_.end()) {
        TRACE(LOCAL, "... not found");
    } else {
        cl_int res;
        map_thread::iterator it = kernelMaps_.find(util::get_thread_id());
        if (it == kernelMaps_.end()) {
            cl_kernel *kernels = new cl_kernel[kernels_.size()];
            res = clCreateKernelsInProgram(program_, kernels_.size(), kernels, NULL);
            ASSERTION(res == CL_SUCCESS);

            map_kernel map;

            for (cl_uint i = 0; i < kernels_.size(); ++i) {
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

            kernelMaps_.insert(map_thread::value_type(util::get_thread_id(), map));
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

void
code_repository::insert(module *m)
{
    push_back(m);
}

kernel_t *
code_repository::get_kernel(gmac_kernel_id_t key)
{
    kernel_t *ret = NULL;
    for (Parent::const_iterator it  = Parent::begin();
                                it != Parent::end();
                              ++it) {
        ret = (*it)->get_kernel(key);
        if (ret != NULL) break;
    }

    return ret;
}

kernel_t *
code_repository::get_kernel(const std::string &name)
{
    kernel_t *ret = NULL;
    for (Parent::const_iterator it  = Parent::begin();
                                it != Parent::end();
                              ++it) {
        ret = (*it)->get_kernel(name);
        if (ret != NULL) break;
    }

    return ret;
}

}}}
