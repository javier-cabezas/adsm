#ifndef GMAC_HAL_CUDA_EVENT_H_
#define GMAC_HAL_CUDA_EVENT_H_

#include <cuda.h>
#include <driver_types.h>
#include <vector_types.h>

#include "trace/logger.h"

#include "util/unique.h"

namespace __impl { namespace hal {

namespace detail {
    class _event;
    class stream;
}
    
namespace cuda {

namespace virt {
    class aspace;
}

typedef util::shared_ptr<hal::detail::_event> hal_event_ptr;
typedef hal::detail::stream hal_stream;

class stream;

class GMAC_LOCAL operation {
    friend class _event_common_t;

    bool synced_;
    CUevent eventStart_;
    CUevent eventEnd_;

    stream &stream_;
    operation(stream &s);

    template <typename R>
    R execute(std::function<R()> f);
    gmacError_t sync();
public:
};

class GMAC_LOCAL _event_common_t :
    public hal::detail::_event {
    friend class code::kernel;
    friend class virt::aspace;
    friend class device;
    friend class list_event;

    typedef hal::detail::_event parent;

#ifdef USE_TRACE
    hal::time_t timeBase_;
#endif
protected:
    typedef std::list<operation> list_operation;
    list_operation operations_;
    list_operation::iterator syncOpBegin_;

    // Not instantiable
    _event_common_t(bool async, parent::type t, virt::aspace &context);

    template <typename R>
    R add_operation(hal_event_ptr ptr, stream &stream, std::function<R()> f);

    //stream &get_stream();
    gmacError_t sync_no_exec();
    gmacError_t sync();

    state get_state();

public:
    void set_barrier(virt::aspace &as, CUstream stream);
};

class GMAC_LOCAL _event_t :
    public _event_common_t {

    friend class virt::aspace;

    typedef _event_common_t parent;

protected:
    virtual void reset(bool async, type t);

    _event_t(bool async, parent::type t, virt::aspace &context);
public:
    virt::aspace &get_vaspace();

    void set_synced();
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
    friend class stream;
    friend class code::kernel;

    //void set_barrier(hal_stream &stream);
public:
    list_event()
    {
    }

    gmacError_t sync();
    void set_synced();

    void add_event(hal_event_ptr event); 

    size_t size() const;
};

}}}

#endif /* GMAC_HAL_CUDA_EVENT_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
