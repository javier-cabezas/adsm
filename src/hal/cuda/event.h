#ifndef GMAC_HAL_CUDA_EVENT_H_
#define GMAC_HAL_CUDA_EVENT_H_

#include <cuda.h>
#include <driver_types.h>
#include <vector_types.h>

#include "hal/types-detail.h"

#include "util/Logger.h"
#include "util/unique.h"

namespace __impl { namespace hal {
    
namespace cuda {

class GMAC_LOCAL _event_common_t {
    friend class context_t;
    friend class device;
    friend class event_t;
    friend class kernel_t;
    stream_t *stream_;

protected:
    CUevent eventStart_;
    CUevent eventEnd_;

    hal::time_t timeBase_;

    // Not instantiable
    _event_common_t();

    void begin(stream_t &stream);
    void end();

    stream_t &get_stream();
};

class GMAC_LOCAL _event_t :
    public hal::detail::_event_t<implementation_traits>,
    public _event_common_t {
    friend class context_t;
    friend class event_t;

    typedef hal::detail::_event_t<implementation_traits> Parent;

protected:
    virtual void reset(bool async, type t);

    _event_t(bool async, Parent::type t, context_t &context);
public:
    gmacError_t sync();

    state get_state();
};

class GMAC_LOCAL event_deleter {
public:
    void operator()(_event_t *ev);
};

class GMAC_LOCAL event_t {
    friend class context_t;
    friend class kernel_t;

private:
    util::shared_ptr<_event_t> ptrEvent_;

    event_t(bool async, _event_t::type t, context_t &context);

public:
    typedef _event_t event_type;
    inline
    event_t()
    {
    }

#ifdef USE_CXX0X
    inline
    event_t(event_t &&event) :
        ptrEvent_(std::move(event.ptrEvent_))
    {
    }
#endif

    inline
    event_t(const event_t &event) :
        ptrEvent_(event.ptrEvent_)
    {
    }

    inline
    event_t &operator=(const event_t &event)
    {
        if (&event != this) {
            ptrEvent_ = event.ptrEvent_;
        }

        return *this;
    }

#ifdef USE_CXX0X
    inline
    event_t &operator=(event_t &&event)
    {
        ptrEvent_ = std::move(event.ptrEvent_);
        return *this;
    }
#endif

    inline
    gmacError_t sync()
    {
        ASSERTION(ptrEvent_);

        gmacError_t ret = ptrEvent_->sync();
        return ret;
    }

    inline
    void begin(stream_t &stream)
    {
        ASSERTION(ptrEvent_);

        ptrEvent_->begin(stream);
    }

    inline
    void end()
    {
        ASSERTION(ptrEvent_);

        ptrEvent_->end();
    }

    inline
    void reset()
    {
        ptrEvent_.reset();
    }

    inline
    bool is_valid() const
    {
        return ptrEvent_;
    }

    template <typename F>
    inline
    void add_trigger(F fun)
    {
        ASSERTION(ptrEvent_);

        ptrEvent_->add_trigger(fun);
    }
};

}}}

#endif /* GMAC_HAL_CUDA_EVENT_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
