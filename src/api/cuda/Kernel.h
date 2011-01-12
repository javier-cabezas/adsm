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
    CUfunction f_;

public:
    Kernel(const core::KernelDescriptor & k, CUmodule mod);
    core::KernelLaunch * launch(core::KernelConfig & c);
};

class GMAC_LOCAL KernelConfig : public core::KernelConfig {
protected:
    dim3 grid_;
    dim3 block_;
    size_t shared_;

    CUstream stream_;
public:
    /// \todo Remove this piece of shit
    KernelConfig(dim3 grid, dim3 block, size_t shared, cudaStream_t tokens, CUstream stream);

    dim3 grid() const { return grid_; }
    dim3 block() const { return block_; }
    size_t shared() const { return shared_; }
};

class GMAC_LOCAL KernelLaunch : public core::KernelLaunch, public cuda::KernelConfig, public util::NonCopyable {
protected:
    // \todo Is this really necessary?
    const Kernel & kernel_;
    CUfunction f_;

    KernelLaunch(const Kernel & k, const KernelConfig & c);
public:

    gmacError_t execute();
    friend class Kernel;
};

}}

#endif
