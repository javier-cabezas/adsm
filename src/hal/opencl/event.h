#ifndef GMAC_HAL_OPENCL_EVENT_H_
#define GMAC_HAL_OPENCL_EVENT_H_

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

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
    public hal::detail::_event_t<implementation_traits>,
    public _event_common_t,
    public gmac::util::mutex<_event_t>,
    public util::unique<_event_t> {

    friend class context_t;
    friend class event_t;

    typedef hal::detail::_event_t<implementation_traits> Parent;

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
    util::shared_ptr<_event_t> ptrEvent_;

    event_t(bool async, _event_t::type t, context_t &context);

public:
    typedef _event_t event_type;

    /**
     * Default constructor. Creates an empty event
     */
    event_t();

#ifdef USE_CXX0X
    /**
     * Move constructor
     *
     * \param event Source event for the move
     */
    event_t(event_t &&event);
#endif
    /**
     * Copy constructor
     *
     * \param event Source event for the copy
     */
    event_t(const event_t &event);

#ifdef USE_CXX0X
    /**
     * Move operator. Moves the given event into this one
     *
     * \param event Source event for the copy
     * \return A reference to the implicit event
     */
    event_t &operator=(event_t &&event);
#endif
    /**
     * Assignment operator. Copies the given event to this one
     *
     * \param event Source event for the copy
     * \return A reference to the implicit event
     */
    event_t &operator=(const event_t &event);

    /**
     * Blocks execution until the event has been completed. The event must be bound to an operation
     *
     * \return The error code of the operation bound to the event
     */
    gmacError_t sync();

    /**
     * Sets the event as completed. The event must be bound to an operation
     */
    void set_synced();

    void begin(stream_t &stream);

    void invalidate();

    /**
     * Tells whether the event must be bound to an operation or not
     *
     * \return A boolean that tells whether the event must be bound to an operation or not
     */
    bool is_valid() const;

    _event_t &operator*();

    cl_event &operator()();

    template <typename F>
    void add_trigger(F fun);
};

}}}

#endif /* GMAC_HAL_OPENCL_EVENT_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
