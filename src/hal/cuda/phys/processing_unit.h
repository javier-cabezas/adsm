#ifndef GMAC_HAL_CUDA_PROCESSING_UNIT_H_
#define GMAC_HAL_CUDA_PROCESSING_UNIT_H_

#include "util/unique.h"

#include "hal/detail/phys/processing_unit.h"

#include "hal/cuda/types.h"

namespace __impl { namespace hal { namespace cuda {
    
namespace virt {
typedef hal::detail::virt::aspace hal_aspace;
class aspace;
}

namespace phys {

typedef hal::detail::phys::processing_unit hal_processing_unit;

class platform;

class GMAC_LOCAL processing_unit :
    public hal_processing_unit,
    public gmac::util::mutex<processing_unit> {

    //friend class aspace;
    friend list_platform hal::phys::get_platforms();

    typedef hal_processing_unit parent;
    typedef gmac::util::mutex<processing_unit> lock;

protected:
    CUdevice cudaDevice_;

    int major_;
    int minor_;

    GmacDeviceInfo info_;
    bool isInfoInitialized_;

public:
    processing_unit(platform &plat, aspace &as, CUdevice cudaDevice);

    hal_stream *create_stream(virt::hal_aspace &as);
    gmacError_t destroy_stream(hal_stream &stream);

    CUdevice get_cuda_id() const;

    int get_major() const;
    int get_minor() const;

    size_t get_total_memory() const;
    size_t get_free_memory() const;

    gmacError_t get_info(GmacDeviceInfo &info);
};

}}}}

#endif /* GMAC_HAL_CUDA_PROCESSING_UNIT_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
