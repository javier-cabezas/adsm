#include "unit/init.h"
#include "unit/core/Process.h"
#include  "core/Mode.h"


using gmac::core::Mode;
using __impl::core::Accelerator;
using __impl::core::ModeMap;
using __impl::core::Process;
using __impl::core::QueueMap;
using namespace __impl::util;
using __impl::core::ThreadQueue;


Mode *ProcessTest::Mode_ = NULL;

void ProcessTest::SetUpTestCase() {
	InitProcess();
	if(Mode_ != NULL) return;
	Mode_ = dynamic_cast<Mode *>(Process::getInstance().createMode(0));
	ASSERT_TRUE(Mode_ != NULL);
	Mode_->initThread();
}


TEST( ModeMapTest,MemberFunc){

	InitProcess();
	Mode *mode_= dynamic_cast<Mode *>(Process::getInstance().createMode(0));
	ASSERT_TRUE(mode_ != NULL);
	//mode_->initThread();
	Accelerator &acc_=GetAccelerator();
	ASSERT_TRUE(&acc_ != NULL);
	ModeMap mm;
//	ASSERT_TRUE(&mm != NULL);
	
	typedef std::map<__impl::core::Mode *, Accelerator *> Parent;
	typedef Parent::iterator iterator;
	std::pair<iterator, bool> ib=mm.insert(mode_,&acc_);
	ASSERT_TRUE(ib.second);
	ASSERT_TRUE(mm.remove(*mode_)==1);
	ASSERT_TRUE(mm.remove(*mode_)==0);


}

TEST_F(ProcessTest,QueueMapTestCase){
	unsigned id=GetThreadId();
	ASSERT_TRUE(id != 0);

	ThreadQueue tq;
	ASSERT_TRUE(tq.queue != NULL);

	typedef std::map<THREAD_T, ThreadQueue *> Parent;
	typedef Parent::iterator iterator;
	QueueMap qm;
	//ASSERT_TRUE(&qm != NULL);
	std::pair<iterator, bool> ib = qm.insert(id, &tq);
	ASSERT_TRUE(ib.second);

	//cteate another Mode,because  want to test member  function  push()
	// can not cteate it , bacause only one accelerator in current OS.
	//Mode *mode_= dynamic_cast<Mode *>(Process::getInstance().createMode(1));
	//ASSERT_TRUE(mode_ != NULL);

	//qm.push(id,*mode_);

	// 因为tq 如果在这释放的话，退出作用域时还会调用析构函数，释放，所以抛出异常
	//ASSERT_ANY_THROW(qm.erase(id));
	//qm.cleanup();

}
