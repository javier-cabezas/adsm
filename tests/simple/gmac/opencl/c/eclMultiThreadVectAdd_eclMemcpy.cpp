#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "gmac/opencl.h"
#include "utils.h"

static unsigned vecSize = 16 * 1024 * 1024;

const char *nIterStr = "GMAC_NITER";
const char *vecSizeStr = "GMAC_VECSIZE";

static const char *kernel = "\
							__kernel void vecAdd(__global float *c, __global float *a, __global float *b)\
							{\
							unsigned i = get_global_id(0);\
							\
							c[i] = a[i] + b[i];\
							}\
							";
static const char *kernel_A = "\
							__kernel void vecAdd_A(__global float *c, __global float *a, __global float *b)\
							{\
							unsigned i = get_global_id(0);\
							\
							c[i] = a[i] + b[i];\
							}\
							";
static float *resultA;
static THREAD_T threadIdA;
static THREAD_T threadIdB;
static THREAD_T threadIdC;
static THREAD_T threadIdD;
static THREAD_T threadIdE;

static int ThreadBody_A()
{
	//assert(eclCompileSource(kernel) == eclSuccess);
//#if 0
	float *a, *b;

	//assert(eclCompileSource(kernel) == eclSuccess);

	fprintf(stdout, "Thread A: Vector: %f\n", 1.0 * vecSize / 1024 / 1024);

 	// Alloc & init input data
 	assert(eclMalloc((void **)&a, vecSize * sizeof(float)) == eclSuccess);
 	assert(eclMalloc((void **)&b, vecSize * sizeof(float)) == eclSuccess);
 	// Alloc output data
 	assert(eclMalloc((void **)&resultA, vecSize * sizeof(float)) == eclSuccess);
	//assert(eclMalloc(input, vecSize * sizeof(float)) == eclSuccess);


 	for(unsigned i = 0; i < vecSize; i++) {
 		a[i] = 1.f;
 		b[i] = 1.f;
 	}

	// Call the kernel
	size_t globalSize = vecSize;
	ecl_kernel kernel;
	assert(eclGetKernel("vecAdd", &kernel) == eclSuccess);
	assert(eclSetKernelArgPtr(kernel, 0, resultA) == eclSuccess);
	assert(eclSetKernelArgPtr(kernel, 1, a) == eclSuccess);
	assert(eclSetKernelArgPtr(kernel, 2, b) == eclSuccess);
	assert(eclCallNDRange(kernel, 1, NULL, &globalSize, NULL) == eclSuccess);

	// Check the result in the CPU
	float error = 0.f;
	for(unsigned i = 0; i < vecSize; i++) {
		error += (resultA[i] - (a[i] + b[i]));
	}
	fprintf(stderr, "Thread A: Error: %f\n", error);

	eclReleaseKernel(kernel);

 	eclFree(a);
	eclFree(b);
//#endif
	return 0;
}

static int ThreadBody_B(void *input)
{
	printf("\r\nThread B: ");
 	for(unsigned i = 0; i < 10; i++) {
 		printf("%f ",(*(float**)(input))[i]);
 	}
	printf("\r\n");

	return 0;
}

static int ThreadBody_C(void *input)
{
	printf("\r\nThread C: ");
	for(unsigned i = 0; i < 10; i++) {
		printf("%f ",(*(float**)(input))[i]);
	}
	printf("\r\n");

	return 0;
}


static int ThreadBody_D(void *input)
{
	printf("\r\nThread D: ");
	for(unsigned i = 0; i < 10; i++) {
		printf("%f ",(*(float**)(input))[i]);
	}
	printf("\r\n");

	return 0;
}

static int ThreadBody_E(void *input)
{
	float* input_E = *(float**)input;
	float *resultB, *temp, *temp_2;

	printf("Thread E: \r\n");
	for(unsigned i = 0; i < 10; i++) {
		printf("%f ",input_E[i]);
	}
	printf("\r\n");
	//  
 	assert(eclCompileSource(kernel_A) == eclSuccess);
 
 	fprintf(stdout, "Thread E: Vector: %f\n", 1.0 * vecSize / 1024 / 1024);
 
  	assert(eclMalloc((void **)&temp, vecSize * sizeof(float)) == eclSuccess);
 	assert(eclMalloc((void **)&resultB, vecSize * sizeof(float)) == eclSuccess);
	assert(eclMalloc((void **)&temp_2, vecSize * sizeof(float)) == eclSuccess);

	eclMemcpy(temp_2, resultA, vecSize * sizeof(float));
 	
	for(unsigned i = 0; i < vecSize; i++) {
 		temp[i] = 1.f; 
 	}
	size_t globalSize = vecSize;
	ecl_kernel kernel;	

	assert(eclGetKernel("vecAdd_A", &kernel) == eclSuccess);
	assert(eclSetKernelArgPtr(kernel, 0, resultB) == eclSuccess);
	assert(eclSetKernelArgPtr(kernel, 1, temp) == eclSuccess);
	//assert(eclSetKernelArgPtr(kernel, 2, resultA) == eclSuccess);
	assert(eclSetKernelArgPtr(kernel, 2, temp_2) == eclSuccess);
	assert(eclCallNDRange(kernel, 1, NULL, &globalSize, NULL) == eclSuccess);
	// 
// 	// Check the result in the CPU
	float error = 0.f;
	for(unsigned i = 0; i < vecSize; i++) {
		error += resultB[i] - (temp[i] + temp_2[i]);	
	}
	printf("resultB: %f ",resultB[0]);
	fprintf(stderr, "Thread E: Error: %f\n", error);

	eclReleaseKernel(kernel);

	eclFree(temp);
	eclFree(resultB);
	eclFree(temp_2);

	return 0;
}

void *addVector(void *ptr)
{
	return NULL;
}

void main(int argc, char *argv[])
{

	/* Thread B C D E wait for thread A to end
	 * Thread B C D read the computation result from thread A
	 * Thread E use the result of thread A to calculate
	 */
	assert(eclCompileSource(kernel) == eclSuccess);

	threadIdA = thread_create(thread_routine(ThreadBody_A),NULL);
	Sleep(10);

	thread_wait(threadIdA);

	threadIdB = thread_create(thread_routine(ThreadBody_B),&resultA);
	threadIdC = thread_create(thread_routine(ThreadBody_C),&resultA);
	threadIdD = thread_create(thread_routine(ThreadBody_D),&resultA);
	threadIdE = thread_create(thread_routine(ThreadBody_E),&resultA);
	thread_wait(threadIdB);
	thread_wait(threadIdC);
	thread_wait(threadIdD);
	thread_wait(threadIdE);

	printf("\r\nmain: ");
	for(unsigned i = 0; i < 10; i++) {
		printf("%f ",resultA[i]);
	}
	printf("\r\n");

	eclFree(resultA);

	system("pause");
}