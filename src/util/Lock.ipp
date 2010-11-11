#ifndef GMAC_UTIL_LOCK_IPP_
#define GMAC_UTIL_LOCK_IPP_

namespace gmac { namespace util {

inline
void ParaverLock::enter() const
{
#ifdef PARAVER
    if(paraver::trace == NULL) return;
    paraver::trace->__pushEvent(*event, id);
#endif
}

inline
void ParaverLock::locked() const
{
#ifdef PARAVER
    if(paraver::trace == NULL) return;
    paraver::trace->__pushState(
        paraver::trace->__pushEvent(*event, 0), *exclusive);
    exclusive_ = true;
#endif
}

inline
void ParaverLock::done() const
{
#ifdef PARAVER
    if(paraver::trace == NULL) return;
    paraver::trace->__pushEvent(*event, 0);
    exclusive_ = false;
#endif
}


inline
void ParaverLock::exit() const
{
#ifdef PARAVER
    if(paraver::trace == NULL) return;
    if(exclusive_) paraver::trace->__popState();
    exclusive_ = false;
#endif
}

}}
#endif
