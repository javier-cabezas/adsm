#ifndef GMAC_HAL_CUDA_EVENT_H_
#define GMAC_HAL_CUDA_EVENT_H_

#include <cuda.h>
#include <driver_types.h>
#include <vector_types.h>

#include "trace/logger.h"

#include "util/unique.h"

namespace __impl { namespace hal {
    
namespace cuda {

namespace virt {
    class aspace;
}

class stream;

class GMAC_LOCAL _event_common_t {
    friend class virt::aspace;
    friend class device;
    friend class kernel;
    friend class kernel_cpu;
    friend class list_event;

    stream *stream_;

protected:
    CUevent eventStart_;
    CUevent eventEnd_;

    hal::time_t timeBase_;

    // Not instantiable
    _event_common_t();

    void begin(stream &stream);
    void end();

    stream &get_stream();
};

class GMAC_LOCAL _event_t :
    public hal::detail::_event,
    public _event_common_t,
    public util::unique<_event_t> {

    friend class virt::aspace;

    typedef hal::detail::_event parent;

protected:
    virtual void reset(bool async, type t);

    _event_t(bool async, parent::type t, virt::aspace &context);
public:
    virt::aspace &get_vaspace();

    gmacError_t sync();

    void set_synced();

    state get_state();
};

class GMAC_LOCAL event_deleter {
public:
    void operator()(_event_t *ev);
};

static
event_ptr
create_event(bool async, _event_t::type t, virt::aspace &as);

typedef hal::detail::event_ptr hal_event_ptr;
typedef hal::detail::list_event list_event_detail;
typedef hal::detail::stream hal_stream;

class GMAC_LOCAL list_event :
    public list_event_detail,
    protected std::list<event_ptr> {
    typedef std::list<event_ptr> parent;

    friend class virt::aspace;
    friend class kernel;

    void set_barrier(hal_stream &stream);
public:
    gmacError_t sync();
    void set_synced();

    void add_event(hal_event_ptr event); 

    size_t size() const;
};

}}}

#endif /* GMAC_HAL_CUDA_EVENT_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
