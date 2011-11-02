#ifndef GMAC_HAL_CUDA_DEVICE_H_
#define GMAC_HAL_CUDA_DEVICE_H_

#include "hal/device.h"
#include "util/unique.h"

#include "types.h"

namespace __impl { namespace hal { namespace cuda {

class coherence_domain;

typedef hal::detail::device<implementation_traits> hal_device;

class device :
    public hal_device,
    public util::unique<device> {
    friend class context_t;

    typedef hal_device Parent;
protected:
    CUdevice cudaDevice_;

    int major_;
    int minor_;

    void set_context(context_t &context);
public:
    device(CUdevice cudaDevice, coherence_domain &coherenceDomain);

    context_t *create_context(const SetSiblings &siblings = None);
    gmacError_t destroy_context(context_t &context);

    stream_t *create_stream(context_t &context);
    gmacError_t destroy_stream(stream_t &stream);

    int get_major() const;
    int get_minor() const;

    size_t get_total_memory() const;
    size_t get_free_memory() const;

    bool has_direct_copy(const Parent &dev) const;
};

}}}

#endif /* GMAC_HAL_DEVICE_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
