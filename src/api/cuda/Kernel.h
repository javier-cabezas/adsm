#ifndef GMAC_API_CUDA_KERNEL_H_
#define GMAC_API_CUDA_KERNEL_H_

#include <cuda.h>
#include <driver_types.h>
#include <vector_types.h>

#include "config/common.h"
#include "core/Kernel.h"

namespace __impl { namespace cuda {

class Mode;

class KernelLaunch;
class KernelConfig;

class GMAC_LOCAL Kernel : public core::Kernel {
    friend class KernelLaunch;
protected:
    CUfunction _f;

public:
    Kernel(const core::KernelDescriptor & k, CUmodule mod);
    core::KernelLaunch * launch(core::KernelConfig & c);
};

class GMAC_LOCAL KernelConfig : public core::KernelConfig {
protected:
    dim3 _grid;
    dim3 _block;
    size_t _shared;

    CUstream _stream;
public:
    /// \todo Remove this piece of shit

    KernelConfig(const KernelConfig & c);
    KernelConfig(dim3 grid, dim3 block, size_t shared, cudaStream_t tokens);

    inline void stream(CUstream s) { _stream = s; }
    dim3 grid() const { return _grid; }
    dim3 block() const { return _block; }
    size_t shared() const { return _shared; }
};

class GMAC_LOCAL KernelLaunch : public core::KernelLaunch, public cuda::KernelConfig {
protected:
    // \todo Is this really necessary?
    const Kernel & _kernel;
    CUfunction _f;

    KernelLaunch(const Kernel & k, const KernelConfig & c);
	KernelLaunch &operator =(const KernelLaunch &);
public:

    gmacError_t execute();
    friend class Kernel;
};

}}

#endif
