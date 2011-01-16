#include "unit/init.h"
#include "api/cuda/Accelerator.h"
#include "gtest/gtest.h"

//added//
#include "api/cuda/Context.h"
#include "api/cuda/Mode.h"
#include "core/Process.h"
////



#include <cuda.h>

using gmac::cuda::Accelerator;

void InitAccelerator()
{
    if(Accelerator_ != NULL) return;
    InitTrace();
    ASSERT_EQ(CUDA_SUCCESS, cuInit(0));
    int count = 0;
    ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetCount(&count));
    ASSERT_GT(count, 0);

    CUdevice dev;
    ASSERT_EQ(CUDA_SUCCESS, cuDeviceGet(&dev, 0));
    Accelerator_ = new Accelerator(dev, 0);
    ASSERT_TRUE(Accelerator_ != NULL);
}




//added//

using __impl::cuda::Mode;
using __impl::core::Process;
using __impl::cuda::Context;


void InitContext()
{
    if(Context_ !=NULL) return;
    InitTrace();
    InitAccelerator();
    Process::create<Process>(); 
    Process &proc =Process::getInstance();
    proc.addAccelerator(*Accelerator_); //*accelerator
    Mode *mode_=dynamic_cast<Mode*>(Process::getInstance().createMode(0)); 
    ASSERT_TRUE(mode_ !=NULL);
    mode_->initThread();
    Accelerator *acc=dynamic_cast<Accelerator*> (Accelerator_);
    ASSERT_TRUE(acc !=NULL); 
    Context_=new Context(*acc,*mode_);
    ASSERT_TRUE(Context_!=NULL); 
} 

  



