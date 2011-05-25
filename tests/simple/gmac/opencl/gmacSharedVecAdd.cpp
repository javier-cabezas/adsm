#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gmac/opencl.h>

#include "utils.h"
#include "debug.h"

const char *nIterStr = "GMAC_NITER";
const char *vecSizeStr = "GMAC_VECSIZE";

const unsigned nIterDefault = 1;
const size_t vecSizeDefault = 1024 * 1024;

unsigned nIter = 0;
unsigned vecSize = 0;
const size_t blockSize = 32;


static float *a, *b;
static struct param {
	int i;
	float *ptr;
    char *prefix;
} *param;

const char *kernel = "\
__kernel void vecAdd(__global float *c, __global const float *a, __global const float *b, unsigned size, unsigned offset)\
{\
    unsigned i = get_global_id(0);\
    if(i >= size) return;\
\
    c[i] = a[i + offset] + b[i + offset];\
}\
";

void *addVector(void *ptr)
{
    static char buffer[1024];

	gmactime_t s, t;
	struct param *p = (struct param *)ptr;
    char *prefix = p->prefix;
	ocl_error ret = oclSuccess;

	ret = oclMalloc((void **)&p->ptr, vecSize * sizeof(float));
	assert(ret == oclSuccess);
	// Call the kernel
	getTime(&s);

    size_t localSize = blockSize;
    size_t globalSize = vecSize / blockSize;
    if(vecSize % blockSize) globalSize++;
    globalSize *= localSize;

    ocl_kernel kernel;

    assert(oclKernelGet("vecAdd", &kernel) == oclSuccess);
    cl_mem tmp = cl_mem(oclPtr(p->ptr));
    assert(oclKernelSetArg(kernel, 0, &tmp, sizeof(cl_mem)) == oclSuccess);
    tmp = cl_mem(oclPtr(a));
    assert(oclKernelSetArg(kernel, 1, &tmp, sizeof(cl_mem)) == oclSuccess);
    tmp = cl_mem(oclPtr(b));
    assert(oclKernelSetArg(kernel, 2, &tmp, sizeof(cl_mem)) == oclSuccess);
    assert(oclKernelSetArg(kernel, 3, &vecSize, sizeof(vecSize)) == oclSuccess);
    unsigned offset = p->i * long(vecSize);
    assert(oclKernelSetArg(kernel, 4, &offset, sizeof(offset)) == oclSuccess);
    assert(oclKernelLaunch(kernel, 1, NULL, &globalSize, &localSize) == oclSuccess);

	getTime(&t);
    snprintf(buffer, 1024, "%s-Run: ", prefix);
	printTime(&s, &t, buffer, "\n");

	getTime(&s);
	float error = 0;
	for(unsigned i = 0; i < vecSize; i++) {
		error += p->ptr[i] - (a[i + p->i * vecSize] + b[i + p->i * vecSize]);
		//error += p->ptr[i] - 1.0f;
	}
	getTime(&t);
    snprintf(buffer, 1024, "%s-CheckFull: ", prefix);
	printTime(&s, &t, buffer, "\n");
    assert(error == 0);

	return NULL;
}

float do_test(GmacGlobalMallocType allocType, const char *prefix)
{
    static char buffer[1024];
	thread_t *nThread;
	unsigned n = 0;
	ocl_error ret = oclSuccess;

	gmactime_t s, t;

	nThread = (thread_t *)malloc(nIter * sizeof(thread_t));
	param = (struct param *)malloc(nIter * sizeof(struct param));

	getTime(&s);
	// Alloc & init input data
	ret = oclGlobalMalloc((void **)&a, nIter * vecSize * sizeof(float), allocType);
	assert(ret == oclSuccess);
	ret = oclGlobalMalloc((void **)&b, nIter * vecSize * sizeof(float), allocType);
	assert(ret == oclSuccess);

	// Alloc output data
	getTime(&t);
    snprintf(buffer, 1024, "%s-Alloc: ", prefix);
	printTime(&s, &t, buffer, "\n");

    getTime(&s);
	valueInit(a, 1.0, nIter * vecSize);
	valueInit(b, 1.0, nIter * vecSize);
    getTime(&t);

    snprintf(buffer, 1024, "%s-Init: ", prefix);
    printTime(&s, &t, buffer, "\n");

	for(n = 0; n < nIter; n++) {
		param[n].i = n;
		param[n].prefix = (char *) prefix;
		nThread[n] = thread_create(addVector, &(param[n]));
	}

	for(n = 0; n < nIter; n++) {
		thread_wait(nThread[n]);
	}

    getTime(&s);
	float error = 0;
	for(n = 0; n < nIter; n++) {
		for(unsigned i = 0; i < vecSize; i++) {
			error += param[n].ptr[i] - 2.f;
		}
	}
    getTime(&t);

    snprintf(buffer, 1024, "%s-Check: ", prefix);
    printTime(&s, &t, buffer, "\n");


    getTime(&s);
    for(n = 0; n < nIter; n++) {
		oclFree(param[n].ptr);
	}

	oclFree(a);
	oclFree(b);

	free(param);
	free(nThread);

    getTime(&t);

    snprintf(buffer, 1024, "%s-Free: ", prefix);
    printTime(&s, &t, buffer, "\n");

    return error;
}


int main(int argc, char *argv[])
{
    assert(oclCompileSource(kernel) == oclSuccess);

	setParam<unsigned>(&nIter, nIterStr, nIterDefault);
	setParam<unsigned>(&vecSize, vecSizeStr, vecSizeDefault);

	vecSize = vecSize / nIter;
	if(vecSize % nIter) vecSize++;

    float error;

    error = do_test(GMAC_GLOBAL_MALLOC_REPLICATED, "Replicated");
    if (error != 0.f) abort();
    error = do_test(GMAC_GLOBAL_MALLOC_CENTRALIZED, "Centralized");
    if (error != 0.f) abort();
    
    return error != 0;
}
