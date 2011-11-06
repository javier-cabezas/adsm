#ifndef GMAC_HAL_OPENCL_EVENT_H_
#define GMAC_HAL_OPENCL_EVENT_H_

#include <CL/cl.h>

#include "hal/types-detail.h"

#include "util/Logger.h"
#include "util/unique.h"

namespace __impl { namespace hal { namespace opencl {

class GMAC_LOCAL _event_common_t {
    friend class device;
    friend class context_t;
    friend class kernel_t;
    stream_t *stream_;

protected:
    cl_event event_;

    hal::time_t timeBase_;

    // Not instantiable
    _event_common_t();

    void begin(stream_t &stream);

    stream_t &get_stream();

public:
    cl_event &operator()()
    {
        return event_;
    }
};

class GMAC_LOCAL _event_t :
    public hal::detail::event_t<device, backend_traits, implementation_traits>,
    public _event_common_t {
    friend class context_t;
    friend class event_t;

    typedef hal::detail::event_t<device, backend_traits, implementation_traits> Parent;

protected:
    virtual void reset(bool async, type t);

    _event_t(bool async, Parent::type t, context_t &context);
public:
    gmacError_t sync();

    void set_synced();

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
    util::smart_ptr<_event_t>::shared ptrEvent_;

    event_t(bool async, _event_t::type t, context_t &context);

public:
    inline
    event_t()
    {
    }

#ifdef USE_CXX0X
    inline
    event_t(event_t &&event) :
        ptrEvent_(std::move(event.ptrEvent_))
    {
        printf("Per move!\n");
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
        printf("= Per move!\n");
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
    void set_synced()
    {
        ASSERTION(ptrEvent_);

        ptrEvent_->set_synced();
    }



    inline
    void begin(stream_t &stream)
    {
        ASSERTION(ptrEvent_);

        ptrEvent_->begin(stream);
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

    inline
    cl_event &operator()()
    {
        return (*ptrEvent_.get())();
    }
};

}}}

#endif /* GMAC_HAL_OPENCL_EVENT_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
