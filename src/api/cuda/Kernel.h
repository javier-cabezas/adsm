#ifndef __API_CUDADRV_KERNEL_H_
#define __API_CUDADRV_KERNEL_H_

#include <kernel/Kernel.h>

#include <cuda.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>

namespace gmac { namespace cuda {

class Mode;

class KernelLaunch;
class KernelConfig;

class Kernel : public gmac::Kernel {
protected:
    CUfunction _f;

public:
    Kernel(const gmac::KernelDescriptor & k, CUmodule mod);
    gmac::KernelLaunch * launch(gmac::KernelConfig & c);

    friend class KernelLaunch;
};

class KernelConfig : public gmac::KernelConfig {
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

class KernelLaunch : public gmac::KernelLaunch, public KernelConfig {
protected:
    // \todo Is this really necessary?
    const Kernel & _kernel;
    CUfunction _f;

    KernelLaunch(const Kernel & k, const KernelConfig & c);
public:

    gmacError_t execute();
    friend class Kernel;
};

}}

#endif
