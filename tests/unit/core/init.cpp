#include "unit/init.h"
#include "trace/Tracer.h"
#include "core/Process.h"
#include "core/Accelerator.h"
#include "core/Context.h" //added 

#include "gtest/gtest.h"

using __impl::core::Accelerator;
using __impl::core::Process;
using gmac::core::Context; 


Accelerator *Accelerator_ = NULL;
Context *Context_=NULL; //added
static bool Trace_ = false;


void InitTrace(void)
{
    if(Trace_ == true) return;
    Trace_ = true;
    gmac::trace::InitTracer();
}

void FiniTrace(void)
{
    if(Trace_ == false) return;
    Trace_ = false;
    gmac::trace::FiniTracer();
}

__impl::core::Accelerator &GetAccelerator()
{
    return *Accelerator_;
}

void FiniAccelerator()
{
    if(Accelerator_ == NULL) return;
    delete Accelerator_;
    Accelerator_ = NULL;
}



gmac::core::Context &GetContext()
{
    return *Context_;
}

void FiniContext()
{
    if(Context_ == NULL) return;
    delete dynamic_cast<__impl::core::Context *>(Context_);
}

void InitProcess()
{
    InitTrace();
    InitAccelerator();

    Process::create<Process>();
    Process &proc = Process::getInstance();
    proc.addAccelerator(*Accelerator_);

    InitContext();
}


void FiniProcess()
{
    FiniContext();
    Process::destroy();
    Accelerator_ = NULL;
    Context_ = NULL;
    FiniTrace();
}

