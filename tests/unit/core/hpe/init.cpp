#include "unit/init.h"
#include "trace/Tracer.h"
#include "core/hpe/Process.h"
#include "core/hpe/Accelerator.h"
#include "core/hpe/Context.h"

#include "gtest/gtest.h"

using __impl::core::hpe::Accelerator;
using gmac::core::hpe::Process;
using gmac::core::hpe::Context; 


Accelerator *Accelerator_ = NULL;
Context *Context_=NULL;

Accelerator &GetAccelerator()
{
    return *Accelerator_;
}

void FiniAccelerator()
{
    if(Accelerator_ == NULL) return;
    delete Accelerator_;
    Accelerator_ = NULL;
}

Context &GetContext()
{
    return *Context_;
}

void FiniContext()
{
    if(Context_ == NULL) return;
    delete Context_;
    Context_ = NULL;
}

void FiniMode()
{
}

void InitModel()
{
    InitAccelerator();
    Process &proc = Process::getInstance<Process &>();
    proc.addAccelerator(*Accelerator_);
}


void FiniModel()
{
    FiniAccelerator();
}

