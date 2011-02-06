#include "Accelerator.h"
#include "core/Accelerator.h"
#include "core/Process.h"
#include "core/Mode.h"

using gmac::core::Mode;
using __impl::core::Process;


//check virtual gmacError_t copyToHost(hostptr_t host, const accptr_t acc,size_t size)=0
//check virtual gmacError_t copyToAccelerator(accptr_t acc, const hostptr_t host, size_t size)=0
TEST_F(AcceleratorTest, AcceleratorMemory) {
    int *buffer = new int[Size_];
    int *canary = new int[Size_];

    memset(buffer, 0xa5, Size_ * sizeof(int));
    memset(canary, 0x5a, Size_ * sizeof(int));
    accptr_t device = NULL;
    ASSERT_TRUE(GetAccelerator().malloc(device, Size_ * sizeof(int)) == gmacSuccess);
    ASSERT_TRUE(device != NULL);
    ASSERT_TRUE(GetAccelerator().copyToAccelerator(device, hostptr_t(buffer), Size_ * sizeof(int)) == gmacSuccess);
    ASSERT_TRUE(GetAccelerator().copyToHost(hostptr_t(canary), device, Size_ * sizeof(int)) == gmacSuccess);
    ASSERT_TRUE(memcmp(buffer, canary, Size_ * sizeof(int)) == 0);  //compare mem size
    ASSERT_TRUE(GetAccelerator().free(device) == gmacSuccess);
    delete[] canary;
    delete[] buffer;
}




//check gmacError_t__impl::core::Accelerator::malloc(accptr_t& addr ,size_t size, unsigned align=1)
// the alignment of mem alloc must be a power of two
TEST_F(AcceleratorTest, AcceleratorAligment) {
    const int max = 32 * 1024 * 1024;
    for(int n = 1; n < max; n <<= 1) {
        accptr_t device = NULL;
        ASSERT_TRUE(GetAccelerator().malloc(device, Size_, n) == gmacSuccess);
        ASSERT_TRUE(device != NULL);
        ASSERT_TRUE(GetAccelerator().free(device) == gmacSuccess);
    }

}

//added
//use Size2_
TEST_F(AcceleratorTest, AcceleratorAligment2) {
    const int max = 32 * 1024 * 1024;
    for(int n = 1; n < max; n <<= 1) {
        accptr_t device = NULL;
        ASSERT_TRUE(GetAccelerator().malloc(device, Size2_, n) == gmacSuccess); 
        ASSERT_TRUE(device != NULL);
        ASSERT_TRUE(GetAccelerator().free(device) == gmacSuccess);
    }

}

//check virtual Mode* createMode(Process &proc)=0 
//check void registerMode(Mode &mode)
//check void unregisterMode(Mode &mode)
TEST_F(AcceleratorTest, CreateMode){

     Process::create<Process>();  
     Process &proc=Process::getInstance();

     ASSERT_TRUE(&proc != NULL);

     ASSERT_TRUE(GetAccelerator().load() == 0) << "the value is :" << GetAccelerator().load(); 
     Mode *mode_ = GetAccelerator().createMode(proc);
     ASSERT_TRUE(mode_!=NULL);
     ASSERT_TRUE(GetAccelerator().load() == 1) << "the value is :" << GetAccelerator().load();
     GetAccelerator().unregisterMode(*mode_);
      
    
     ASSERT_TRUE(GetAccelerator().load() == 0) << "the value is :" << GetAccelerator().load(); 
     GetAccelerator().registerMode(*mode_);
     ASSERT_TRUE(GetAccelerator().load() == 1) << "the value is :" << GetAccelerator().load(); 

     delete dynamic_cast<__impl::core::Mode *>(mode_);
     ASSERT_TRUE(GetAccelerator().load() == 0) << "the value is :" << GetAccelerator().load(); 

     Process::destroy();
}





