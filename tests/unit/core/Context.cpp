#include "Context.h"
#include "core/Context.h"
#include "core/Accelerator.h"





TEST_F(ContextTest,ContextMemory){
	
	int *buffer = new int[Size_];
	int *canary = new int[Size_];

	memset(buffer, 0xa5, Size_ * sizeof(int));
	memset(canary, 0x5a, Size_ * sizeof(int));

	accptr_t device = NULL;
	ASSERT_TRUE(GetAccelerator().malloc(device, Size_ * sizeof(int)) == gmacSuccess);// device 
	ASSERT_TRUE(device != NULL);
	ASSERT_TRUE(GetContext().copyToAccelerator(device, hostptr_t(buffer), Size_ * sizeof(int)) == gmacSuccess);
	ASSERT_TRUE(memcmp(buffer, canary, Size_ * sizeof(int)) != 0);
	ASSERT_TRUE(GetContext().copyToHost(hostptr_t(canary), device, Size_ * sizeof(int)) == gmacSuccess);
	ASSERT_TRUE(memcmp(buffer, canary, Size_ * sizeof(int)) == 0);


	memset(buffer, 0, Size_ * sizeof(int));
	memset(canary, 0, Size_ * sizeof(int));
	ASSERT_TRUE(memcmp(buffer,canary,Size_ * sizeof(int)) == 0);

	accptr_t dstdevice=NULL;
	ASSERT_TRUE(GetAccelerator().malloc(dstdevice,Size_*sizeof(int))==gmacSuccess); //device
	ASSERT_TRUE(dstdevice != NULL);
	ASSERT_TRUE(GetContext().copyToHost(hostptr_t(buffer),device,Size_ * sizeof(int))==gmacSuccess);
	ASSERT_TRUE(memcmp(buffer, canary, Size_ * sizeof(int)) != 0);
	ASSERT_TRUE(GetContext().copyAccelerator(dstdevice,device,Size_*sizeof(int))==gmacSuccess);
	ASSERT_TRUE(GetContext().copyToHost(hostptr_t(canary),dstdevice,Size_ * sizeof(int))==gmacSuccess);
	ASSERT_TRUE(memcmp(buffer,canary,Size_ * sizeof(int)) == 0);

	ASSERT_TRUE(GetAccelerator().free(device) == gmacSuccess);
	ASSERT_TRUE(GetAccelerator().free(dstdevice) == gmacSuccess);
	delete[] canary;
	delete[] buffer;
	

}
////


