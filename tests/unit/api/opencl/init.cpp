#include "unit/init.h"
//#include "api/opencl/Accelerator.h"
#include "gtest/gtest.h"

#include <CL/cl.h>

//using gmac::opencl::Accelerator;

void InitAccelerator()
{
    if(Accelerator_ != NULL) return;
    InitTrace();
    // TODO: Create OpenCL accelerator -- pending
}
