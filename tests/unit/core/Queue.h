#ifndef TEST_GMAC_CORE_QUEUE_H_
#define TEST_GMAC_CORE_QUEUE_H_

#include "unit/init.h"
#include "gtest/gtest.h"

#include "core/Queue.h"
#include "core/Mode.h"

using gmac::core::Mode;

class QueueTest:public testing::Test{

  public:
     static gmac::core::Mode *Mode_;
     // const static size_t Size_= 4* 1024*1024; 
     static void SetUpTestCase();
     static void TearDownTestCase()
     {
        Mode_->detach();
        FiniProcess();
        Mode_=NULL;
     }


};
#endif
