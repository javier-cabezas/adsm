#include "unit/init.h"
#include "unit/core/Queue.h"


#include "core/Process.h"
#include "core/Mode.h"

#include "gtest/gtest.h"

using gmac::core::Mode;
using __impl::core::Queue;
using __impl::core::Process;
using __impl::core::ThreadQueue;

Mode *QueueTest::Mode_=NULL;

void QueueTest::SetUpTestCase()
{
     InitProcess();
     if (Mode_!=NULL) return;
     Mode_=dynamic_cast<Mode *> (Process::getInstance().createMode(0));
     ASSERT_TRUE(Mode_!=NULL);
     Mode_->initThread();
}
TEST_F(QueueTest,MemberFun)
{
    ThreadQueue temp_; 
    ASSERT_TRUE(temp_.queue!=NULL);
    temp_.queue->push(Mode_);
    Mode *last = dynamic_cast<__dbc::core::Mode *>(temp_.queue->pop());
    ASSERT_EQ(Mode_,last);
    //Queue actual_("ThreadQueue");
    //ASSERT_EQ(*temp_.queue,actual_);
}





