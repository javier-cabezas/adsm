#include "unit/core/hpe/Process.h"
#include "core/hpe/Mode.h"
#include "core/hpe/Process.h"


using __impl::core::hpe::Accelerator;
using __impl::core::hpe::Mode;
using __impl::core::hpe::ModeMap;
using gmac::core::hpe::Process;

extern void CUDA(Process &);
extern void OpenCL(Process &);

Process *ProcessTest::createProcess()
{
    Process *proc = new Process();
    if(proc == NULL) return proc;
#if defined(USE_CUDA)
    CUDA(*proc);
#endif
#if defined(USE_OPENCL)
    OpenCL(*proc);
#endif
    return proc;
}


TEST_F(ProcessTest, ModeMap) {

    Process *proc = createProcess();
    ASSERT_TRUE(proc != NULL);

	Mode *mode = proc->createMode(0);
	ASSERT_TRUE(mode != NULL);

	ModeMap mm;
	typedef std::map<__impl::core::hpe::Mode *, Accelerator *> Parent;
	typedef Parent::iterator iterator;
	std::pair<iterator, bool> ib = mm.insert(mode, &mode->getAccelerator());
	ASSERT_TRUE(ib.second);
	ASSERT_TRUE(mm.remove(*mode) == 1);
	ASSERT_TRUE(mm.remove(*mode) == 0);

    proc->destroy();
}
