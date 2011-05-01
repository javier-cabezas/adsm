#include "Context.h"
#include "unit/core/hpe/Context.h"
#include "core/hpe/Context.h"
#include "core/hpe/Mode.h"

using gmac::core::hpe::Process;
using __impl::core::hpe::Mode;
using __impl::opencl::hpe::Context;

Process *ContextTest::Process_ = NULL;

extern void OpenCL(Process &);

void ContextTest::SetUpTestCase() {
    Process_ = new Process();
#if defined(USE_OPENCL)
    OpenCL(*Process_);
#endif
}

void ContextTest::TearDownTestCase() {
    delete Process_;
    Process_ = NULL;
}


TEST_F(OpenCLContextTest, ContextMemory){
	

    unsigned count = Process_->nAccelerators();
    for(unsigned i = 0; i < count; i++) {
        Mode *mode = Process_->createMode(i);
        ASSERT_TRUE(mode != NULL);

        Context *ctx = new Context(mode->getAccelerator(), *mode);
        ASSERT_TRUE(ctx != NULL);

        ContextMemory(*mode, *ctx)

        delete ctx; ctx = NULL;
        mode->detach(); mode = NULL;
    }
}


