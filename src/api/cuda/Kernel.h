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

class KernelConfig;
class KernelLaunch;

class GMAC_LOCAL Argument : public util::ReusableObject<Argument> {
	friend class Kernel;
    const void * ptr_;
    size_t size_;
    long_t offset_;
public:
    Argument(const void * ptr, size_t size, long_t offset);

    const void * ptr() const { return ptr_; }
    size_t size() const { return size_; }
    long_t offset() const { return offset_; }
};



class GMAC_LOCAL Kernel : public core::Kernel {
    friend class KernelLaunch;
protected:
    CUfunction f_;

public:
    Kernel(const core::KernelDescriptor & k, CUmodule mod);
    KernelLaunch * launch(KernelConfig & c);
};

typedef std::vector<Argument> ArgsVector;

class GMAC_LOCAL KernelConfig : public ArgsVector {
protected:
    static const unsigned StackSize_ = 4096;

    uint8_t stack_[StackSize_];
    size_t argsSize_;

    dim3 grid_;
    dim3 block_;
    size_t shared_;

    CUstream stream_;

    KernelConfig(const KernelConfig & c);
public:
    /// \todo create a pool of objects to avoid mallocs/frees
    KernelConfig();
    KernelConfig(dim3 grid, dim3 block, size_t shared, cudaStream_t tokens, CUstream stream);

    void pushArgument(const void * arg, size_t size, long_t offset);

    size_t argsSize() const;
    uint8_t *argsArray();

    dim3 grid() const { return grid_; }
    dim3 block() const { return block_; }
    size_t shared() const { return shared_; }
};

class GMAC_LOCAL KernelLaunch : public core::KernelLaunch,
                                public KernelConfig,
                                public util::NonCopyable {
    friend class Kernel;

protected:
    // \todo Is this really necessary?
    const Kernel & kernel_;
    CUfunction f_;

    KernelLaunch(const Kernel & k, const KernelConfig & c);
public:

    gmacError_t execute();
};

}}

#include "Kernel-impl.h"

#endif
