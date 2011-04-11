#ifndef TEST_GMAC_CORE_QUEUE_H_
#define TEST_GMAC_CORE_QUEUE_H_

#include "unit/init.h"
#include "gtest/gtest.h"

#include "core/hpe/Queue.h"
#include "core/hpe/Mode.h"

using gmac::core::hpe::Mode;

class QueueTest:public testing::Test{

  public:
     static gmac::core::hpe::Mode *Mode_;

     static void SetUpTestCase();
     static void TearDownTestCase()
     {
        Mode_->detach();
        FiniProcess();
        Mode_=NULL;
     }


};
#endif
