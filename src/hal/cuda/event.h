#ifndef GMAC_HAL_CUDA_EVENT_H_
#define GMAC_HAL_CUDA_EVENT_H_

#include <cuda.h>
#include <driver_types.h>
#include <vector_types.h>

#include "hal/types-detail.h"

#include "trace/logger.h"

#include "util/unique.h"

namespace __impl { namespace hal {
    
namespace cuda {

class GMAC_LOCAL _event_common_t {
    friend class aspace;
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

    friend class aspace;

    typedef hal::detail::_event parent;

protected:
    virtual void reset(bool async, type t);

    _event_t(bool async, parent::type t, aspace &context);
public:
    aspace &get_aspace();

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
create_event(bool async, _event_t::type t, aspace &as);

#if 0
class GMAC_LOCAL event_ptr {
    friend class aspace;
    friend class kernel;

private:
    util::shared_ptr<_event_t> ptrEvent_;

    event_ptr(bool async, _event_t::type t, aspace &context);

    inline
    void reset()
    {
        ptrEvent_.reset();
    }
public:
    typedef _event_t event_type;
    typedef _event_t::type type;

    inline
    event_ptr()
    {
    }

#ifdef USE_CXX0X
    inline
    event_ptr(event_ptr &&event) :
        ptrEvent_(std::move(event.ptrEvent_))
    {
    }
#endif

    inline
    event_ptr(const event_ptr &event) :
        ptrEvent_(event.ptrEvent_)
    {
    }

    inline
    event_ptr &operator=(const event_ptr &event)
    {
        if (&event != this) {
            ptrEvent_ = event.ptrEvent_;
        }

        return *this;
    }

#ifdef USE_CXX0X
    inline
    event_ptr &operator=(event_ptr &&event)
    {
        ptrEvent_ = std::move(event.ptrEvent_);
        return *this;
    }
#endif

    inline
    operator bool() const
    {
        return bool(ptrEvent_);
    }

    _event_t *operator->()
    {
        return ptrEvent_.get();
    }

    _event_t &operator*()
    {
        return *ptrEvent_.get();
    }

    template <typename F>
    inline
    void add_trigger(F fun)
    {
        ASSERTION(bool(ptrEvent_));

        ptrEvent_->add_trigger(fun);
    }
};
#endif

typedef hal::detail::event_ptr hal_event_ptr;
typedef hal::detail::list_event list_event_detail;
typedef hal::detail::stream hal_stream;

class GMAC_LOCAL list_event :
    public list_event_detail,
    protected std::list<event_ptr> {
    typedef std::list<event_ptr> parent;

    friend class aspace;
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
