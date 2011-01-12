#ifndef GMAC_API_CUDA_KERNEL_H_
#define GMAC_API_CUDA_KERNEL_H_

#include <cuda.h>
#include <driver_types.h>
#include <vector_types.h>

#include "config/common.h"
#include "core/Kernel.h"
#include "util/NonCopyable.h"

namespace __impl { namespace cuda {

class Mode;

class KernelLaunch;
class KernelConfig;

class GMAC_LOCAL Kernel : public core::Kernel {
    friend class KernelLaunch;
protected:
    cl_kernel f_;

public:
    Kernel(const core::KernelDescriptor & k, cl_program program);
    core::KernelLaunch * launch(core::KernelConfig & c);
};

class GMAC_LOCAL KernelConfig : public core::KernelConfig {
protected:
    cl_uint work_dim_;
    size_t *global_work_offset_;
    size_t *global_work_size_;
    size_t *local_work_size_;

    cl_command_queue stream_;
public:
    /// \todo Remove this piece of shit

    KernelConfig(cl_uint work_dim, size_t *global_work_offset, size_t *global_work_size, size_t *local_work_size);
    ~KernelConfig();

    cl_uint work_dim() const { return work_dim_; }
    size_t *global_work_offset() const { return global_work_offset_; }
    size_t *global_work_size() const { return global_work_size__; }
    size_t *local_work_size() const { return local_work_size_; }
};

class GMAC_LOCAL KernelLaunch : public core::KernelLaunch, public cuda::KernelConfig, public util::NonCopyable {
protected:
    cl_kernel f_;

    KernelLaunch(const Kernel & k, const KernelConfig & c);
public:
    gmacError_t execute();
    friend class Kernel;
};

}}

#endif
