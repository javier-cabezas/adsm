#ifndef GMAC_UTIL_LOCK_IMPL_H_
#define GMAC_UTIL_LOCK_IMPL_H_

namespace gmac { namespace util {

inline
void __Lock::enter() const
{
#ifdef PARAVER
    if(paraver::trace == NULL) return;
    paraver::trace->__pushEvent(*event, id);
#endif
}

inline
void __Lock::locked() const
{
#ifdef PARAVER
    if(paraver::trace == NULL) return;
    paraver::trace->__pushState(
        paraver::trace->__pushEvent(*event, 0), *exclusive);
    exclusive_ = true;
#endif
}

inline
void __Lock::done() const
{
#ifdef PARAVER
    if(paraver::trace == NULL) return;
    paraver::trace->__pushEvent(*event, 0);
    exclusive_ = false;
#endif
}


inline
void __Lock::exit() const
{
#ifdef PARAVER
    if(paraver::trace == NULL) return;
    if(exclusive_) paraver::trace->__popState();
    exclusive_ = false;
#endif
}

}}
#endif
