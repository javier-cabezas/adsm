#include "unit/memory/Object.h"
#include <iostream>

using namespace std;


Protocol* ObjectTest::Protocol_=NULL;
Object* ObjectTest::Object_=NULL;



TEST_F(ObjectTest,SizeFuncTest)
{
	hostptr_t addr_=NULL,end_ = NULL;
	addr_=Object_->addr();
	ASSERT_TRUE(addr_ != NULL);
	end_=Object_->end();
	ASSERT_TRUE(end_ != NULL);
	size_t range=end_ - addr_;
	ASSERT_GT(range,0U);
	size_t size=Object_->size();
	ASSERT_GT(size,0U);
	ASSERT_EQ(range,size);
}


TEST_F(ObjectTest,BlockSizeFuncTest)
{
	for(int i=0;i<Size_;i++)
	{	
		size_t begin_=Object_->blockBase(i);
		size_t en_=Object_->blockEnd(i);
		size_t size=Object_->blockSize();
		ASSERT_EQ(size,en_-begin_)<<"en_:"<<en_\
			<<"begin_:"<<begin_<<"size :"<<size<<endl;

	}
	ASSERT_TRUE(Object_->valid());
}



TEST_F(ObjectTest,OwnerFuncTest)
{
	hostptr_t addr_=NULL,end_ = NULL;
	addr_=Object_->addr();
	ASSERT_TRUE(addr_ != NULL);
 	end_=Object_->end();
 	ASSERT_TRUE(end_ != NULL);

	Mode &mode_=Mode::getCurrent();
	Mode *om_=&Object_->owner(addr_);
	ASSERT_TRUE(om_!= NULL);
 	Mode *om2_=&Object_->owner(--end_);
 	ASSERT_TRUE(om2_ != NULL);

	ASSERT_EQ(&mode_ ,om_);
	ASSERT_EQ(om_,om2_);

	
	ASSERT_TRUE(Object_->removeOwner(*om_) == gmacSuccess);
	ASSERT_TRUE(Object_->removeOwner(*om2_) == gmacSuccess);
	//ASSERT_TRUE(Object_->addOwner(mode_) == gmacSuccess);

// 	om_=&Object_->owner(addr_);
//  	ASSERT_NE(om2_ , om_);

}
