#include "Context.h"
#include "core/Context.h"
#include "core/Accelerator.h"





TEST_F(ContextTest,ContextMemory){
	
	int *buffer  = new int[Size_];
	int *bufferExt = new int[Size_];
	int *canary  = new int[Size_];

	memset(buffer, 0xa5, Size_ * sizeof(int));
	memset(canary, 0x5a, Size_ * sizeof(int));

	accptr_t device(0);
	ASSERT_TRUE(GetAccelerator().map(device, hostptr_t(buffer), Size_ * sizeof(int)) == gmacSuccess);// device 
	ASSERT_TRUE(device != 0);
	ASSERT_TRUE(GetContext().copyToAccelerator(device, hostptr_t(buffer), Size_ * sizeof(int)) == gmacSuccess);
	ASSERT_TRUE(memcmp(buffer, canary, Size_ * sizeof(int)) != 0);
	ASSERT_TRUE(GetContext().copyToHost(hostptr_t(canary), device, Size_ * sizeof(int)) == gmacSuccess);
	ASSERT_TRUE(memcmp(buffer, canary, Size_ * sizeof(int)) == 0);


	memset(buffer,  0, Size_ * sizeof(int));
	memset(bufferExt, 0, Size_ * sizeof(int));
	memset(canary,  0, Size_ * sizeof(int));
	ASSERT_TRUE(memcmp(buffer,canary,Size_ * sizeof(int)) == 0);

	accptr_t dstDevice(0);
	ASSERT_TRUE(GetAccelerator().map(dstDevice, hostptr_t(bufferExt), Size_*sizeof(int)) == gmacSuccess); //device
	ASSERT_TRUE(dstDevice != 0);
	ASSERT_TRUE(GetContext().copyToHost(hostptr_t(buffer), device, Size_ * sizeof(int)) == gmacSuccess);
	ASSERT_TRUE(memcmp(buffer, canary, Size_ * sizeof(int)) != 0);
	ASSERT_TRUE(GetContext().copyAccelerator(dstDevice, device, Size_*sizeof(int)) == gmacSuccess);
	ASSERT_TRUE(GetContext().copyToHost(hostptr_t(canary), dstDevice, Size_ * sizeof(int)) == gmacSuccess);
	ASSERT_TRUE(memcmp(buffer, canary, Size_ * sizeof(int)) == 0);

	ASSERT_TRUE(GetAccelerator().unmap(hostptr_t(buffer),  Size_ * sizeof(int)) == gmacSuccess);
	ASSERT_TRUE(GetAccelerator().unmap(hostptr_t(bufferExt), Size_ * sizeof(int)) == gmacSuccess);
	delete[] canary;
	delete[] buffer;
	delete[] bufferExt;
}
////


