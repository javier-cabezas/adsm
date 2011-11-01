#ifndef GMAC_HAL_TYPES_EVENT_H_
#define GMAC_HAL_TYPES_EVENT_H_

namespace __impl { namespace hal {

namespace detail {

template <typename D, typename B, typename I>
class GMAC_LOCAL event_t {
    friend class I::context;
    friend class I::kernel;

    typedef typename I::context context_parent_t;
public:
    enum type {
        Transfer,
        Kernel
    };

private:
    context_parent_t &context_;
    bool isAsynchronous_;
    bool synced_;
    type type_;

protected:
    gmacError_t err_;

    hal::time_t timeQueued_;
    hal::time_t timeSubmit_;
    hal::time_t timeStart_;
    hal::time_t timeEnd_;

    event_t(type t, context_parent_t &context);

public:
    enum state {
        Queued,
        Submit,
        Start,
        End
    };

    virtual gmacError_t sync();
    virtual bool is_synced() const;

    context_parent_t &get_context();

    type get_type() const;
    virtual state get_state();

    hal::time_t get_time_queued() const;
    hal::time_t get_time_submit() const;
    hal::time_t get_time_start() const;
    hal::time_t get_time_end() const;
};

template <typename D, typename B, typename I>
class GMAC_LOCAL async_event_t {
    typedef typename I::context context_parent_t;

    typedef event_t<D, B, I> Parent;

private:
    bool synced_;

protected:
    typename Parent::state state_;

    async_event_t();

    void set_synced(bool synced);

public:
    bool is_synced() const;
};

}

}}

#endif /* EVENT_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
