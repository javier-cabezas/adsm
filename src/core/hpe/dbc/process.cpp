#ifdef USE_DBC

#include "process.h"	

#include "core/hpe/process.h"

namespace __dbc { namespace core { namespace hpe {

process::process() :
    __impl::core::hpe::process()
{
}

process::~process()
{
}

void
process::initThread(bool userThread, THREAD_T parent)
{
   __impl::core::hpe::process::initThread(userThread, parent);
}

void
process::finiThread(bool userThread)
{
   __impl::core::hpe::process::finiThread(userThread);
}

#if 0
__impl::core::hpe::Mode*
process::createMode(int acc)
{
    return  __impl::core::hpe::process::createMode(acc);
}

void
process::removeMode(__impl::core::hpe::Mode &mode)
{
    __impl::core::hpe::process::removeMode(mode);
}
#endif

}}}
#endif


