#include "types.h"

namespace __impl { namespace hal { namespace opencl {

void
_event_t::reset(bool async, type t)
{
    // Locking is not needed
    async_ = async;
    type_ = t;
    err_ = gmacSuccess;
    synced_ = false;
    state_ = None;
    cl_int res = clReleaseEvent(event_);
    ASSERTION(res == CL_SUCCESS);
    
    remove_triggers();
}

_event_t::state
_event_t::get_state()
{
    lock_write();

    if (state_ != End) {
        cl_int status;
        cl_int res = clGetEventInfo(event_,
                                    CL_EVENT_COMMAND_EXECUTION_STATUS,
                                    sizeof(cl_int),
                                    &status, NULL);
        if (res == CL_SUCCESS) {
            if (status == CL_QUEUED) {
                state_ = Queued;
            } else if (status == CL_SUBMITTED) {
                state_ = Submit;
            } else if (status == CL_RUNNING) {
                state_ = Start;
            } else if (status == CL_COMPLETE) {
                state_ = End;
                synced_ = true;
            } else {
                FATAL("Unhandled value");
            }
        }
    }

    unlock();

    return state_;
}

list_event::~list_event()
{
    for (Parent::iterator it  = Parent::begin();
                          it != Parent::end();
                          it++) {
        cl_int res = clReleaseEvent((**it)());
        ASSERTION(res == CL_SUCCESS);
    }
}

gmacError_t
list_event::sync()
{
    unsigned nevents;
    cl_event *evs = get_event_array(nevents);

    cl_int res = CL_SUCCESS;
    if (nevents != 0) {
        res = clWaitForEvents(Parent::size(), evs);
    }
    set_synced();

    return error(res);
}

void
list_event::set_synced()
{
    for (Parent::iterator it  = Parent::begin();
            it != Parent::end();
            it++) {
        (**it).set_synced();
    }
}

cl_event *
list_event::get_event_array(stream_t &stream, unsigned &nevents)
{
    cl_event *ret = NULL;
    if (Parent::size() > 0) {
        nevents = 0;
        for (Parent::iterator it = Parent::begin();
             it != Parent::end();
             it++) {
            locker::lock_read(**it);
            if (((*it)->get_stream().get_print_id() != stream.get_print_id()) &&
                !(*it)->is_synced()) {
                nevents++;
            }
        }

        ret = new cl_event[nevents];

        unsigned i = 0;
        for (Parent::iterator it = Parent::begin();
             it != Parent::end();
             it++) {
            if (((*it)->get_stream().get_print_id() != stream.get_print_id()) &&
                !(*it)->is_synced()) {
                ret[i++] = (**it)();
            }
            locker::unlock(**it);
        }
        ASSERTION(i == nevents);

        printf("a) Waiting for %u events\n", nevents);
    }

    return ret;
}

cl_event *
list_event::get_event_array(unsigned &nevents)
{
    cl_event *ret = NULL;
    if (Parent::size() > 0) {
        nevents = 0;
        for (Parent::iterator it = Parent::begin();
             it != Parent::end();
             it++) {
            locker::lock_read(**it);
            if (!(*it)->is_synced()) {
                nevents++;
            }
        }

        ret = new cl_event[nevents];

        unsigned i = 0;
        for (Parent::iterator it = Parent::begin();
             it != Parent::end();
             it++) {
            if (!(*it)->is_synced()) {
                ret[i++] = (**it)();
            }
            locker::unlock(**it);
        }
        ASSERTION(i == nevents);

        printf("b) Waiting for %u events\n", nevents);
    }

    return ret;
}

size_t
list_event::size() const
{
    return Parent::size(); 
}

void
list_event::add_event(event_t event) 
{
    locker::lock_read(**event);
    if (std::find(begin(), end(), *event) != end()) {
        cl_event ev = event();
        cl_int res = clRetainEvent(ev);
        ASSERTION(res == CL_SUCCESS);
        Parent::push_back(*event);
    }
    locker::unlock(**event);
}

}}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
