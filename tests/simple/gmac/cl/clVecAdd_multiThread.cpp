
/* two thread: shared platform, shared device, shared context, shared command_queue
 * BUT: seperate data collection
 *
 *
 */
/******************************************************************************************************************
 *						RESULT
 *	1. Threads with separate platforms, devices, contexts, command_queues: works well (actually No shared data, No syn. operations)
 *	2. Threads with shared platform, but separate devices, contexts, command_queues: works well (actually No shared data, No syn. operations)
 *	3. Threads with shared platform, devices, but separate contexts, command_queues: works well (actually No shared data, No syn. operations)
 *	4. Threads with shared platform, devices, but separate contexts, command_queues: works well (actually No shared data, No syn. operations)
 *
 *
 *
 */

//#ifdef GMAC_ENABLE
#if 1
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cassert>

#include "gmac/cl.h"
#include "utils.h"
#include "debug.h"

static const char *vecSizeStr = "GMAC_VECSIZE";
static const unsigned vecSizeDefault = 16 * 1024 * 1024;
static unsigned vecSize = vecSizeDefault;
static const char *msg = "Done!";

static const char *kernel_source = "\
							__kernel void vecAdd(__global float *c, __global const float *a, __global const float *b, unsigned size)\
							{\
							unsigned i = get_global_id(0);\
							if(i >= size) return;\
							\
							c[i] = a[i] + b[i];\
							}\
							";

typedef struct __OpenCLEnv
{
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue command_queue;
} OpenCLEnv;

static OpenCLEnv openCLEnv = {NULL,NULL,NULL,NULL};

//#define SHARED_CONTEXT
//#define SHARED_CAMMAND_QUEUE



void Thread_A()
{
	cl_platform_id platform;
	cl_device_id device;
	cl_int error_code;
	cl_context context;
	cl_command_queue command_queue;
	cl_program program;
	cl_kernel kernel;
	float *a, *b, *c;
	gmactime_t s, t;

	//error_code = clGetPlatformIDs(1, &platform, NULL);
	//assert(error_code == CL_SUCCESS);
	platform = openCLEnv.platform;
	//error_code = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
	//assert(error_code == CL_SUCCESS);
	device = openCLEnv.device;

#ifndef SHARED_CONTEXT
	context = clCreateContext(0, 1, &device, NULL, NULL, &error_code);
	assert(error_code == CL_SUCCESS);
#else
	context = openCLEnv.context;
#endif

#ifndef SHARED_CAMMAND_QUEUE
	command_queue = clCreateCommandQueue(context, device, 0, &error_code);
	assert(error_code == CL_SUCCESS);
#else
	command_queue = openCLEnv.command_queue;
#endif

	setParam<unsigned>(&vecSize, vecSizeStr, vecSizeDefault);              ///////
	fprintf(stdout, "Thread A: Vector: %f\n", 1.0 * vecSize / 1024 / 1024);

	program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &error_code);
	assert(error_code == CL_SUCCESS);
	error_code = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	assert(error_code == CL_SUCCESS);
	kernel = clCreateKernel(program, "vecAdd", &error_code);
	assert(error_code == CL_SUCCESS);

	getTime(&s);

	error_code = clMalloc(command_queue, (void **)&a, vecSize * sizeof(float));
	assert(error_code == CL_SUCCESS);
	error_code = clMalloc(command_queue, (void **)&b, vecSize * sizeof(float));
	assert(error_code == CL_SUCCESS);
	error_code = clMalloc(command_queue, (void **)&c, vecSize * sizeof(float));
	assert(error_code == CL_SUCCESS);
	getTime(&t);

	printTime(&s, &t, "Thread A: Alloc: ", "\n");

	float sum = 0.f;
	getTime(&s);
	valueInit(a, 1.1f, vecSize);
	valueInit(b, 1.f, vecSize);
	getTime(&t);
	printTime(&s, &t, "Thread A: Init: ", "\n");

	for(unsigned i = 0; i < vecSize; i++) {
		sum += a[i] + b[i];
	}

	getTime(&s);
	size_t global_size = vecSize;

	cl_mem c_device = clGetBuffer(context, c);
	error_code = clSetKernelArg(kernel, 0, sizeof(cl_mem), &c_device);
	assert(error_code == CL_SUCCESS);
	cl_mem a_device = clGetBuffer(context, a);
	error_code = clSetKernelArg(kernel, 1, sizeof(cl_mem), &a_device);
	assert(error_code == CL_SUCCESS);
	cl_mem b_device = clGetBuffer(context, b);
	error_code = clSetKernelArg(kernel, 2, sizeof(cl_mem), &b_device);
	assert(error_code == CL_SUCCESS);
	error_code = clSetKernelArg(kernel, 3, sizeof(vecSize), &vecSize);
	assert(error_code == CL_SUCCESS);

	error_code  = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
	assert(error_code == CL_SUCCESS);
	error_code = clFinish(command_queue);
	assert(error_code == CL_SUCCESS);

	printf("Thread A:¡¡C[0]: %f\n",c[0]);

	getTime(&t);
	printTime(&s, &t, "Thread A: Run: ", "\n");

	getTime(&s);
	float error = 0.f;
	float check = 0.f;
	for(unsigned i = 0; i < vecSize; i++) {
		error += c[i] - (a[i] + b[i]);
		check += c[i];
	}
	printf("Thread A Check: %f\n",error);

	if (sum != check) {
		printf("Thread A: Sum: %f vs %f\n", sum, check);
		abort();
	}

	printf("\r\n Thread A clFree\r\n");
	clFree(command_queue, a);
	printf("\r\n Thread A clFree out 1\r\n");
	clFree(command_queue, b);
	printf("\r\n Thread A clFree out 2\r\n");
	clFree(command_queue, c);
	printf("\r\n Thread A clFree out 3 OVER\r\n");
}

void Thread_B()
{
	cl_platform_id platform;
	cl_device_id device;
	cl_int error_code;
	cl_context context;
	cl_command_queue command_queue;
	cl_program program;
	cl_kernel kernel;
	float *a, *b, *c;
	gmactime_t s, t;

	//error_code = clGetPlatformIDs(1, &platform, NULL);
	//assert(error_code == CL_SUCCESS);
	platform = openCLEnv.platform;
	//error_code = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
	//assert(error_code == CL_SUCCESS);
	device = openCLEnv.device;
#ifndef SHARED_CONTEXT
	context = clCreateContext(0, 1, &device, NULL, NULL, &error_code);
	assert(error_code == CL_SUCCESS);
#else
	context = openCLEnv.context;
#endif
#ifndef SHARED_CAMMAND_QUEUE
	command_queue = clCreateCommandQueue(context, device, 0, &error_code);
	assert(error_code == CL_SUCCESS);
#else
	command_queue = openCLEnv.command_queue;
#endif

	setParam<unsigned>(&vecSize, vecSizeStr, vecSizeDefault);              ///////
	fprintf(stdout, "Thread B: Vector: %f\n", 1.0 * vecSize / 1024 / 1024);

	program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &error_code);
	assert(error_code == CL_SUCCESS);
	error_code = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	assert(error_code == CL_SUCCESS);
	kernel = clCreateKernel(program, "vecAdd", &error_code);
	assert(error_code == CL_SUCCESS);

	getTime(&s);

	error_code = clMalloc(command_queue, (void **)&a, vecSize * sizeof(float));
	assert(error_code == CL_SUCCESS);
	error_code = clMalloc(command_queue, (void **)&b, vecSize * sizeof(float));
	assert(error_code == CL_SUCCESS);
	error_code = clMalloc(command_queue, (void **)&c, vecSize * sizeof(float));
	assert(error_code == CL_SUCCESS);
	getTime(&t);

	printTime(&s, &t, "Thread B: Alloc: ", "\n");

	float sum = 0.f;
	getTime(&s);
	valueInit(a, 1.1f, vecSize);
	valueInit(b, 1.f, vecSize);
	getTime(&t);
	printTime(&s, &t, "Thread B: Init: ", "\n");

	for(unsigned i = 0; i < vecSize; i++) {
		sum += a[i] + b[i];
	}

	getTime(&s);
	size_t global_size = vecSize;

	cl_mem c_device = clGetBuffer(context, c);
	error_code = clSetKernelArg(kernel, 0, sizeof(cl_mem), &c_device);
	assert(error_code == CL_SUCCESS);
	cl_mem a_device = clGetBuffer(context, a);
	error_code = clSetKernelArg(kernel, 1, sizeof(cl_mem), &a_device);
	assert(error_code == CL_SUCCESS);
	cl_mem b_device = clGetBuffer(context, b);
	error_code = clSetKernelArg(kernel, 2, sizeof(cl_mem), &b_device);
	assert(error_code == CL_SUCCESS);
	error_code = clSetKernelArg(kernel, 3, sizeof(vecSize), &vecSize);
	assert(error_code == CL_SUCCESS);

	error_code  = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
	assert(error_code == CL_SUCCESS);
	error_code = clFinish(command_queue);
	assert(error_code == CL_SUCCESS);

	printf("Thread B:¡¡C[0]: %f\n",c[0]);

	getTime(&t);
	printTime(&s, &t, "Thread B: Run: ", "\n");

	getTime(&s);
	float error = 0.f;
	float check = 0.f;
	for(unsigned i = 0; i < vecSize; i++) {
		error += c[i] - (a[i] + b[i]);
		check += c[i];
	}
	printf("Thread B Check: %f\n",error);

	if (sum != check) {
		printf("Thread B: Sum: %f vs %f\n", sum, check);
		abort();
	}
	printf("\r\n Thread B clFree\r\n");
	clFree(command_queue, a);
	printf("\r\n Thread B clFree out 1\r\n");
	clFree(command_queue, b);
	printf("\r\n Thread B clFree out 2\r\n");
	clFree(command_queue, c);
	printf("\r\n Thread B clFree out 3 OVER\r\n");

}


void Thread_C()
{
	cl_platform_id platform;
	cl_device_id device;
	cl_int error_code;
	cl_context context;
	cl_command_queue command_queue;
	cl_program program;
	cl_kernel kernel;
	float *a, *b, *c;
	gmactime_t s, t;

	//error_code = clGetPlatformIDs(1, &platform, NULL);
	//assert(error_code == CL_SUCCESS);
	platform = openCLEnv.platform;
	//error_code = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
	//assert(error_code == CL_SUCCESS);
	device = openCLEnv.device;
#ifndef SHARED_CONTEXT
	context = clCreateContext(0, 1, &device, NULL, NULL, &error_code);
	assert(error_code == CL_SUCCESS);
#else
	context = openCLEnv.context;
#endif
#ifndef SHARED_CAMMAND_QUEUE
	command_queue = clCreateCommandQueue(context, device, 0, &error_code);
	assert(error_code == CL_SUCCESS);
#else
	command_queue = openCLEnv.command_queue;
#endif

	setParam<unsigned>(&vecSize, vecSizeStr, vecSizeDefault);              ///////
	fprintf(stdout, "Thread C: Vector: %f\n", 1.0 * vecSize / 1024 / 1024);

	program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &error_code);
	assert(error_code == CL_SUCCESS);
	error_code = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	assert(error_code == CL_SUCCESS);
	kernel = clCreateKernel(program, "vecAdd", &error_code);
	assert(error_code == CL_SUCCESS);

	getTime(&s);

	error_code = clMalloc(command_queue, (void **)&a, vecSize * sizeof(float));
	assert(error_code == CL_SUCCESS);
	error_code = clMalloc(command_queue, (void **)&b, vecSize * sizeof(float));
	assert(error_code == CL_SUCCESS);
	error_code = clMalloc(command_queue, (void **)&c, vecSize * sizeof(float));
	assert(error_code == CL_SUCCESS);
	getTime(&t);

	printTime(&s, &t, "Thread C: Alloc: ", "\n");

	float sum = 0.f;
	getTime(&s);
	valueInit(a, 1.1f, vecSize);
	valueInit(b, 1.f, vecSize);
	getTime(&t);
	printTime(&s, &t, "Thread C: Init: ", "\n");

	for(unsigned i = 0; i < vecSize; i++) {
		sum += a[i] + b[i];
	}

	getTime(&s);
	size_t global_size = vecSize;

	cl_mem c_device = clGetBuffer(context, c);
	error_code = clSetKernelArg(kernel, 0, sizeof(cl_mem), &c_device);
	assert(error_code == CL_SUCCESS);
	cl_mem a_device = clGetBuffer(context, a);
	error_code = clSetKernelArg(kernel, 1, sizeof(cl_mem), &a_device);
	assert(error_code == CL_SUCCESS);
	cl_mem b_device = clGetBuffer(context, b);
	error_code = clSetKernelArg(kernel, 2, sizeof(cl_mem), &b_device);
	assert(error_code == CL_SUCCESS);
	error_code = clSetKernelArg(kernel, 3, sizeof(vecSize), &vecSize);
	assert(error_code == CL_SUCCESS);

	error_code  = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
	assert(error_code == CL_SUCCESS);
	error_code = clFinish(command_queue);
	assert(error_code == CL_SUCCESS);

	printf("Thread C:¡¡c[0]: %f\n",c[0]);

	getTime(&t);
	printTime(&s, &t, "Thread C: Run: ", "\n");

	getTime(&s);
	float error = 0.f;
	float check = 0.f;
	for(unsigned i = 0; i < vecSize; i++) {
		error += c[i] - (a[i] + b[i]);
		check += c[i];
	}
	printf("Thread C Check: %f\n",error);

	if (sum != check) {
		printf("Thread C: Sum: %f vs %f\n", sum, check);
		abort();
	}

	printf("\r\n Thread C clFree\r\n");
	clFree(command_queue, a);
	printf("\r\n Thread C clFree out 1\r\n");
	clFree(command_queue, b);
	printf("\r\n Thread C clFree out 2\r\n");
	clFree(command_queue, c);
	printf("\r\n Thread C clFree out 3 OVER\r\n");
}


void  main(int argc, char *argv[])
{
	cl_int error_code;

	/***********************************************************************************************/
	/*		Initialize the environment													     		*/
	/***********************************************************************************************/
	error_code = clGetPlatformIDs(1, &(openCLEnv.platform), NULL);
	assert(error_code == CL_SUCCESS);
	error_code = clGetDeviceIDs(openCLEnv.platform, CL_DEVICE_TYPE_GPU, 1, &(openCLEnv.device), NULL);
	assert(error_code == CL_SUCCESS);
	openCLEnv.context = clCreateContext(0, 1, &(openCLEnv.device), NULL, NULL, &error_code);
	assert(error_code == CL_SUCCESS);
	openCLEnv.command_queue = clCreateCommandQueue(openCLEnv.context, openCLEnv.device, 0, &error_code);
	assert(error_code == CL_SUCCESS);
	cl_float* temp;
	clMalloc(openCLEnv.command_queue,(void**)&temp, sizeof(cl_float));

	/************************************************************************/
	/* Invoke the threads                                                    */
	/************************************************************************/
	
	//HANDLE ht1 = CreateThread(NULL,NULL,LPTHREAD_START_ROUTINE(Thread_A),NULL,NULL,NULL);
	//HANDLE ht2 = CreateThread(NULL,NULL,LPTHREAD_START_ROUTINE(Thread_B),NULL,NULL,NULL);
	//HANDLE ht3 = CreateThread(NULL,NULL,LPTHREAD_START_ROUTINE(Thread_C),NULL,NULL,NULL);
	thread_t thread1 = thread_create(thread_routine(Thread_A), NULL);
	//thread_t thread2 = thread_create(thread_routine(Thread_B), NULL);
	//thread_t thread3 = thread_create(thread_routine(Thread_C), NULL);
	
	/************************************************************************/
	/* Waiting for ending of the thread                                     */
	/************************************************************************/
	thread_wait(thread1);
	//thread_wait(thread2);
	//thread_wait(thread3);	
	//WaitForSingleObject(ht3,INFINITE);
	system("pause");

}
#endif
