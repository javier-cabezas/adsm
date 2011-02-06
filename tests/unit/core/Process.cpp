#include "unit/init.h"
#include "unit/core/Process.h"
#include  "core/Mode.h"


using gmac::core::Mode;
using __impl::core::Accelerator;
using __impl::core::ModeMap;
using __impl::core::Process;
using __impl::core::QueueMap;
using __impl::core::ThreadQueue;


TEST_F(ProcessTest, ModeMap) {

	Accelerator &acc_ = GetAccelerator();
	ASSERT_TRUE(&acc_ != NULL);

    Mode *mode = NULL;
	mode = dynamic_cast<Mode *>(Process::getInstance().createMode(0));
	ASSERT_TRUE(mode != NULL);
	mode->initThread();

	ModeMap mm;
	typedef std::map<__impl::core::Mode *, Accelerator *> Parent;
	typedef Parent::iterator iterator;
	std::pair<iterator, bool> ib = mm.insert(mode, &acc_);
	ASSERT_TRUE(ib.second);
	ASSERT_TRUE(mm.remove(*mode) == 1);
	ASSERT_TRUE(mm.remove(*mode) == 0);

    mode->detach();
    delete mode;
}
