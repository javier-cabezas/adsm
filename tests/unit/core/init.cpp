#include "unit/init.h"
#include "trace/Tracer.h"
#include "core/Process.h"
#include "core/Accelerator.h"

#include "gtest/gtest.h"

using __impl::core::Accelerator;
using __impl::core::Process;

Accelerator *Accelerator_ = NULL;
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

void InitProcess()
{
    InitTrace();
    InitAccelerator();

    Process::create<Process>();
    Process &proc = Process::getInstance();
    proc.addAccelerator(*Accelerator_);
}


void FiniProcess()
{
    ASSERT_TRUE(Accelerator_ != NULL);
    Accelerator_ = NULL;
    Process::destroy();
    FiniTrace();
}

