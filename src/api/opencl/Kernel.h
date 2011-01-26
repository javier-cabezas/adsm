#ifndef GMAC_API_OPENCL_KERNEL_H_
#define GMAC_API_OPENCL_KERNEL_H_

#include <CL/cl.h>

#include <list>

#include "config/common.h"
#include "core/Kernel.h"
#include "util/NonCopyable.h"

namespace __impl { namespace opencl {

class Mode;

class KernelConfig;
class KernelLaunch;

class GMAC_LOCAL Argument : public util::ReusableObject<Argument> {
	friend class Kernel;
    const void * ptr_;
    size_t size_;
    unsigned index_;
public:
    Argument(const void * ptr, size_t size, unsigned index);

    const void * ptr() const { return ptr_; }
    size_t size() const { return size_; }
    unsigned index() const { return index_; }
};



class GMAC_LOCAL Kernel : public core::Kernel {
    friend class KernelLaunch;
protected:
    cl_kernel f_;
public:
    Kernel(const core::KernelDescriptor & k, cl_kernel kernel);
    ~Kernel();
    KernelLaunch * launch(KernelConfig & c);
};

typedef std::list<Argument> ArgsList;

class GMAC_LOCAL KernelConfig : protected ArgsList {
protected:
    static const unsigned StackSize_ = 4096;

    uint8_t stack_[StackSize_];
    size_t argsSize_;

    cl_uint workDim_;
    size_t *globalWorkOffset_;
    size_t *globalWorkSize_;
    size_t *localWorkSize_;

    cl_command_queue stream_;

    KernelConfig(const KernelConfig &config);
public:
    /// \todo Remove this piece of shit
    KernelConfig();
    KernelConfig(cl_uint work_dim, size_t *globalWorkOffset, size_t *globalWorkSize, size_t *localWorkSize, cl_command_queue stream);
    ~KernelConfig();

    void setArgument(const void * arg, size_t size, unsigned index);

    KernelConfig &operator=(const KernelConfig &config);

    cl_uint workDim() const { return workDim_; }
    size_t *globalWorkOffset() const { return globalWorkOffset_; }
    size_t *globalWorkSize() const { return globalWorkSize_; }
    size_t *localWorkSize() const { return localWorkSize_; }
};

class GMAC_LOCAL KernelLaunch : public core::KernelLaunch, public KernelConfig, public util::NonCopyable {
    friend class Kernel;

protected:
    cl_kernel f_;

    KernelLaunch(const Kernel & k, const KernelConfig & c);
public:
    ~KernelLaunch();
    gmacError_t execute();
};

}}

#include "Kernel-impl.h"

#endif
