#include "init.h"

#include "core/Process.h"
#include "core/Mode.h"

void InitModel();
void FiniModel();

__impl::core::Mode *Mode_ = NULL;

using __impl::core::Process;

void InitProcess()
{
    InitModel();
}

void FiniProcess()
{
    FiniModel();
    Process::destroy();
}

__impl::core::Mode &GetMode()
{
    return *Mode_;
}


