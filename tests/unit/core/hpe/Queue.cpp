#include "unit/init.h"
#include "unit/core/hpe/Queue.h"


#include "core/hpe/Process.h"
#include "core/hpe/Mode.h"

#include "gtest/gtest.h"

using gmac::core::hpe::Mode;
using __impl::core::hpe::Queue;
using __impl::core::hpe::Process;
using __impl::core::hpe::ThreadQueue;

Mode *QueueTest::Mode_ = NULL;

void QueueTest::SetUpTestCase()
{
     InitProcess();
     if(Mode_ != NULL) return;
     Mode_ = dynamic_cast<Mode *>(Process::getInstance<Process &>().createMode(0));
     ASSERT_TRUE(Mode_ != NULL);
     Mode_->initThread();
}
TEST_F(QueueTest,MemberFun)
{
    ThreadQueue temp_; 
    ASSERT_TRUE(temp_.queue != NULL);
    temp_.queue->push(Mode_);
    Mode *last = dynamic_cast<gmac::core::hpe::Mode *>(temp_.queue->pop());
    ASSERT_EQ(Mode_,last);
    //Queue actual_("ThreadQueue");
    //ASSERT_EQ(*temp_.queue,actual_);
}





