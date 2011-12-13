#ifndef GMAC_HAL_OPENCL_EVENT_H_
#define GMAC_HAL_OPENCL_EVENT_H_

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "hal/types-detail.h"

#include "trace/logger.h"

#include "util/lock.h"
#include "util/unique.h"

namespace __impl { namespace hal { namespace opencl {

class GMAC_LOCAL _event_common_t {
    friend class context_t;
    friend class device;
    friend class event_ptr;
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
    public util::unique<_event_t> {

    friend class context_t;
    friend class list_event;

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

class GMAC_LOCAL event_ptr {
    friend class context_t;
    friend class kernel_t;
    friend class list_event;

private:
    util::shared_ptr<_event_t> ptrEvent_;

    event_ptr(bool async, _event_t::type t, context_t &context);

    inline
    void reset()
    {
        ptrEvent_.reset();
    }

    inline
    util::shared_ptr<_event_t>
    get_shared_ptr()
    {
        return ptrEvent_;
    }

public:
    typedef _event_t event_type;
    typedef _event_t::type type;

    /**
     * Default constructor. Creates an empty event
     */
    event_ptr();

#ifdef USE_CXX0X
    /**
     * Move constructor
     *
     * \param event Source event for the move
     */
    event_ptr(event_ptr &&event);
#endif
    /**
     * Copy constructor
     *
     * \param event Source event for the copy
     */
    event_ptr(const event_ptr &event);

    /**
     * Assignment operator. Copies the given event to this one
     *
     * \param event Source event for the copy
     * \return A reference to the implicit event
     */
    event_ptr &operator=(const event_ptr &event);

#ifdef USE_CXX0X
    /**
     * Move operator. Moves the given event into this one
     *
     * \param event Source event for the copy
     * \return A reference to the implicit event
     */
    event_ptr &operator=(event_ptr &&event);
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
    void add_trigger(F fun);
};

typedef hal::detail::list_event<implementation_traits> list_event_detail;

class GMAC_LOCAL list_event :
    public list_event_detail,
    protected std::list<util::shared_ptr<_event_t> >,
    public util::locker_rw<hal::detail::_event_t<implementation_traits> > {
    typedef std::list<util::shared_ptr<_event_t> > Parent;

    friend class context_t;
    friend class kernel_t;

protected:
    typedef util::locker_rw<hal::detail::_event_t<implementation_traits> > locker;
    void set_synced();

    cl_event *get_event_array(stream_t &stream, unsigned &nevents);
    cl_event *get_event_array(unsigned &nevents);
public:
    ~list_event();

    gmacError_t sync();

    size_t size() const;

    void add_event(event_ptr event);
};

}}}

#endif /* GMAC_HAL_OPENCL_EVENT_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
