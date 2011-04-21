#include "unit/init.h"
#include "unit/core/hpe/Process.h"
#include "core/hpe/Mode.h"


using gmac::core::hpe::Mode;
using __impl::core::hpe::Accelerator;
using __impl::core::hpe::ModeMap;
using __impl::core::hpe::Process;
using __impl::core::hpe::QueueMap;
using __impl::core::hpe::ThreadQueue;


TEST_F(ProcessTest, ModeMap) {

	Accelerator &acc_ = GetAccelerator();
	ASSERT_TRUE(&acc_ != NULL);

    Mode *mode = NULL;
	Process &proc = dynamic_cast<Process &>(__impl::core::Process::getInstance());
	mode = dynamic_cast<Mode *>(proc.createMode(0));
	ASSERT_TRUE(mode != NULL);
	mode->initThread();

	ModeMap mm;
	typedef std::map<__impl::core::hpe::Mode *, Accelerator *> Parent;
	typedef Parent::iterator iterator;
	std::pair<iterator, bool> ib = mm.insert(mode, &acc_);
	ASSERT_TRUE(ib.second);
	ASSERT_TRUE(mm.remove(*mode) == 1);
	ASSERT_TRUE(mm.remove(*mode) == 0);

    mode->detach();
    delete mode;
}
