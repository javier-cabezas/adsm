#ifndef GMAC_UTIL_DELAYED_EXEC_H_
#define GMAC_UTIL_DELAYED_EXEC_H_

#include "tr1/functional"

namespace __impl { namespace util {

class GMAC_LOCAL functor_iface {
public:
    virtual void operator()() = 0;
};

template <typename F, typename R>
class GMAC_LOCAL functor :
    public functor_iface {
private:
    F fun_;
    R ret_;
public:
    functor(F fun, R ret) :
        fun_(fun),
        ret_(ret)
    {
    }

    void operator()()
    {
        R ret = fun_();
        ASSERTION(ret == ret_);
    }
};

template <typename F>
class GMAC_LOCAL functor<F, void> :
    public functor_iface {
private:
    F fun_;
public:
    functor(F fun) :
        fun_(fun)
    {
    }

    void operator()()
    {
        fun_();
    }
};

using namespace std::tr1;

#define do_member(f,o,...) bind(std::tr1::mem_fn(&f), o, __VA_ARGS__)
#define do_func(f,...)     bind(&f, __VA_ARGS__)

class GMAC_LOCAL delayed_exec {
protected:
    typedef std::list<functor_iface *> list_trigger;
    list_trigger triggers_;

    void exec_triggers()
    {
        list_trigger::iterator it;
        for (it  = triggers_.begin();
             it != triggers_.end();
             it++) {
            (**it)();
        }
    }

    void remove_triggers()
    {
        list_trigger::iterator it;
        for (it  = triggers_.begin();
             it != triggers_.end();
             it++) {
            delete *it;
        }
        triggers_.clear();
    }

    virtual ~delayed_exec()
    {
        remove_triggers();
    }
public:
    template <typename F, typename R>
    void add_trigger(F fun, R ret)
    {
        triggers_.push_back(new functor<F, R>(fun, ret));
    }

    template <typename F>
    void add_trigger(F fun)
    {
        triggers_.push_back(new functor<F, void>(fun));
    }
};

}}

#endif // GMAC_UTIL_DELAYED_EXEC_H_

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
