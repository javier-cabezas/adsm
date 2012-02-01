#ifdef USE_DBC

#include "hpe/core/process.h"

namespace __dbc { namespace core { namespace hpe {

process::process() :
    parent()
{
}

process::~process()
{
}

void
process::init_thread(bool userThread, THREAD_T parent)
{
   parent::init_thread(userThread, parent);
}

void
process::fini_thread(bool userThread)
{
   parent::fini_thread(userThread);
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


