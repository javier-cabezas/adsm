#ifndef USE_MULTI_CONTEXT

#include "Accelerator.h"
#include "core/hpe/Accelerator.h"
#include "core/hpe/Process.h"
#include "core/hpe/Thread.h"
#include "core/hpe/Mode.h"

using gmac::core::hpe::Process;
using gmac::core::hpe::Thread;

using __impl::core::hpe::Mode;
using __impl::core::hpe::Accelerator;

Process *AcceleratorTest::Process_ = NULL;
std::vector<Accelerator *> Accelerators_;

extern void OpenCL(Process &);
extern void CUDA(Process &);

void AcceleratorTest::SetUpTestCase()
{
    Process_ = new Process();
#if defined(USE_CUDA)
    CUDA(*Process_);
#endif
#if defined(USE_OPENCL)
    OpenCL(*Process_);
#endif
}

void AcceleratorTest::TearDownTestCase()
{
    if(Process_ != NULL) Process_->destroy();
    Process_ = NULL;
}

TEST_F(AcceleratorTest, Memory) {
    int *buffer = new int[Size_];
    int *canary = new int[Size_];

    memset(buffer, 0xa5, Size_ * sizeof(int));
    memset(canary, 0x5a, Size_ * sizeof(int));
    accptr_t device(0);
    size_t count = Process_->nAccelerators();
    for(unsigned n = 0; n < count; n++) {
        Accelerator *acc = Process_->getAccelerator(n);
        ASSERT_TRUE(acc != NULL);
        ASSERT_TRUE(acc->map(device, hostptr_t(buffer), Size_ * sizeof(int)) == gmacSuccess);
        ASSERT_TRUE(device != 0);
        ASSERT_TRUE(acc->copyToAccelerator(device, hostptr_t(buffer), Size_ * sizeof(int), Thread::getCurrentMode()) == gmacSuccess);
        ASSERT_TRUE(acc->copyToHost(hostptr_t(canary), device, Size_ * sizeof(int), Thread::getCurrentMode()) == gmacSuccess);
        ASSERT_TRUE(memcmp(buffer, canary, Size_ * sizeof(int)) == 0);  //compare mem size
        ASSERT_TRUE(acc->unmap(hostptr_t(buffer), Size_ * sizeof(int)) == gmacSuccess);
    }
    delete[] canary;
    delete[] buffer;
}

TEST_F(AcceleratorTest, Aligment) {
    const hostptr_t fakePtr = (uint8_t *) 0xcafebabe;
    const int max = 32 * 1024 * 1024;
    size_t count = Process_->nAccelerators();
    for(unsigned i = 0; i < count; i++) {
        Accelerator *acc = Process_->getAccelerator(i);
        ASSERT_TRUE(acc != NULL);
        for(int n = 1; n < max; n <<= 1) {
            accptr_t device(0);
            ASSERT_TRUE(acc->map(device, fakePtr, Size_, n) == gmacSuccess);
            ASSERT_TRUE(device != 0);
            ASSERT_TRUE(acc->unmap(fakePtr, Size_) == gmacSuccess);
        }
    }

}


TEST_F(AcceleratorTest, CreateMode) {

    size_t n = Process_->nAccelerators();
    for(unsigned i = 0; i < n; i++) {
        Accelerator *acc = Process_->getAccelerator(i);
        ASSERT_TRUE(acc != NULL);
        unsigned load = acc->load();
        Mode *mode = acc->createMode(*Process_);
        ASSERT_TRUE(mode != NULL);
        ASSERT_TRUE(acc->load() == load + 1);

        mode->decRef();
        ASSERT_TRUE(acc->load() == load);
    }
}





#endif
