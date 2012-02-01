#ifndef GMAC_UTIL_TRIGGER_H_
#define GMAC_UTIL_TRIGGER_H_

#ifdef USE_CXX0X
#include <functional>
#include <type_traits>
#else
#include <tr1/functional>
#include <tr1/type_traits>
#endif

namespace __impl { namespace util {

#ifdef USE_CXX0X
using std::bind;
#else
using namespace std::tr1;
#endif

#define do_member(f,o,...) util::bind(std::mem_fn(&f), o, __VA_ARGS__)
#define do_func(f,...)     util::bind(&f, __VA_ARGS__)

template <typename T>
class GMAC_LOCAL functor_iface {
public:
    virtual void operator()(T &t) = 0;
};

template <>
class GMAC_LOCAL functor_iface<void> {
public:
    virtual void operator()() = 0;
};

template <typename F, typename T, typename R>
class GMAC_LOCAL functor :
    public functor_iface<T> {
private:
    F fun_;
    R ret_;

public:
    functor(F fun, R ret) :
        fun_(fun),
        ret_(ret)
    {
    }

    void
    operator()(T &t)
    {
        R ret = fun_(t);
        ASSERTION(ret == ret_);
    }
};

template <typename F, typename R>
class GMAC_LOCAL functor<F, void, R> :
    public functor_iface<void> {
private:
    F fun_;
    R ret_;

public:
    functor(F fun, R ret) :
        fun_(fun),
        ret_(ret)
    {
    }

    void
    operator()()
    {
        R ret = fun_();
        ASSERTION(ret == ret_);
    }
};

template <typename F, typename T>
class GMAC_LOCAL functor<F, T, void> :
    public functor_iface<T> {
private:
    F fun_;

public:
    functor(F fun) :
        fun_(fun)
    {
    }

    void
    operator()(T &t)
    {
        fun_(t);
    }
};

template <typename F>
class GMAC_LOCAL functor<F, void, void> :
    public functor_iface<void> {
private:
    F fun_;

public:
    functor(F fun) :
        fun_(fun)
    {
    }

    void
    operator()()
    {
        fun_();
    }
};

template <typename T = void>
class GMAC_LOCAL list_trigger {
    template <class T1> friend class on_construction;
    template <class T1> friend class on_destruction;

protected:
    typedef std::list<functor_iface<T> *> base_list_trigger;
    base_list_trigger triggers_;

    void
    exec_triggers(bool freeTriggers)
    {
        typename base_list_trigger::iterator it;
        for (it  = triggers_.begin();
             it != triggers_.end();
             it++) {
            (**it)();
        }

        if (freeTriggers) {
            remove_triggers();
        }
    }

    template <typename T2 = T>
    void
    exec_triggers(bool freeTriggers, T2 &arg)
    {
        typename base_list_trigger::iterator it;
        for (it  = triggers_.begin();
             it != triggers_.end();
             it++) {
             (**it)(arg);
        }

        if (freeTriggers) {
            remove_triggers();
        }
    }

    void remove_triggers()
    {
        typename base_list_trigger::iterator it;
        for (it  = triggers_.begin();
             it != triggers_.end();
             it++) {
            delete *it;
        }
        triggers_.clear();
    }

    
public:
    inline
    virtual ~list_trigger()
    {
        remove_triggers();
    }

    template <typename F, typename R>
    void add_trigger(F fun, R ret)
    {
        triggers_.push_back(new functor<F, T, R>(fun, ret));
    }

    template <typename F>
    void add_trigger(F fun)
    {
        triggers_.push_back(new functor<F, T, void>(fun));
    }
};

template <typename T>
class GMAC_LOCAL on_construction {
    static list_trigger<T> constructors_;

public:
    on_construction()
    {
        on_construction<T>::constructors_.exec_triggers(false, *reinterpret_cast<T *>(this));
    }

    template <typename F>
    static void add_constructor(F fun)
    {
        constructors_.add_trigger(fun);
    }

    static void fini()
    {
        constructors_.remove_triggers();
    }
};

template <typename T>
list_trigger<T> on_construction<T>::constructors_;

template <typename T>
class GMAC_LOCAL on_destruction {
    static list_trigger<T> destructors_;

public:
    ~on_destruction()
    {
        on_destruction<T>::destructors_.exec_triggers(false, *reinterpret_cast<T *>(this));
    }

    template <typename F>
    static void add_destructor(F fun)
    {
        destructors_.add_trigger(fun);
    }

    static void fini()
    {
        destructors_.remove_triggers();
    }
};

template <typename T>
list_trigger<T> on_destruction<T>::destructors_;


}}

#endif // GMAC_UTIL_TRIGGER_H_

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
