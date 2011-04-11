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
    std::cerr << "Process created" << std::endl;
}

void FiniProcess()
{
    FiniModel();
    Process::destroy();
    std::cerr << "Process destroyed" << std::endl;
}

__impl::core::Mode &GetMode()
{
    return *Mode_;
}


