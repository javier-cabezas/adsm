#ifdef USE_DBC

//#include "core/dbc/Process.h"	
#include "core/Process.h"
#include "core/Mode.h"

namespace __dbc { namespace core {

Process::Process()
{
}
Process::~Process()
{
}
void Process::initThread()
{
   __impl::core::Process::initThread();
}
void Process::finiThread()
{
   __impl::core::Process::finiThread();
}
Mode *Process::createMode(int acc)
{
    return  __impl::core::Process::createMode(acc);
}
void Process::removeMode(__impl::core::Mode& mode)
{
    REQUIRES(&mode != NULL);
    __impl::core::Process::removeMode(mode);
}

}}
#endif


