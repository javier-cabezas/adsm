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
	gmactime_t s, t;
	struct param *p = (struct param *)ptr;
	oclError_t ret = gmacSuccess;

	ret = oclMalloc((void **)&p->ptr, vecSize * sizeof(float));
	assert(ret == gmacSuccess);

	// Call the kernel
	getTime(&s);

    size_t localSize = blockSize;
    size_t globalSize = vecSize / blockSize;
    if(vecSize % blockSize) globalSize++;
    globalSize *= localSize;

    assert(__oclConfigureCall(1, NULL, &globalSize, &localSize) == gmacSuccess);
    cl_mem tmp = cl_mem(oclPtr(p->ptr));
    __oclSetArgument(&tmp, sizeof(cl_mem), 0);
    tmp = cl_mem(oclPtr(a));
    __oclSetArgument(&tmp, sizeof(cl_mem), 1);
    tmp = cl_mem(oclPtr(b));
    __oclSetArgument(&tmp, sizeof(cl_mem), 2);
    __oclSetArgument(&vecSize, sizeof(vecSize), 3);
    unsigned offset = p->i * long(vecSize);
    __oclSetArgument(&offset, sizeof(offset), 4);
    assert(__oclLaunch("vecAdd") == gmacSuccess);
    assert(oclThreadSynchronize() == gmacSuccess);

	getTime(&t);
	printTime(&s, &t, "Run: ", "\n");

	getTime(&s);
	float error = 0;
	for(unsigned i = 0; i < vecSize; i++) {
		error += p->ptr[i] - (a[i + p->i * vecSize] + b[i + p->i * vecSize]);
		//error += (a[i] - b[i]);
	}
	getTime(&t);
	printTime(&s, &t, "Check: ", "\n");
	fprintf(stdout, "Error: %.02f\n", error);

    //assert(error == 0);

	return NULL;
}


int main(int argc, char *argv[])
{
	thread_t *nThread;
	unsigned n = 0;
	oclError_t ret = gmacSuccess;
	gmactime_t s, t;

    assert(__oclPrepareCLCode(kernel) == gmacSuccess);

	setParam<unsigned>(&nIter, nIterStr, nIterDefault);
	setParam<unsigned>(&vecSize, vecSizeStr, vecSizeDefault);

	vecSize = vecSize / nIter;
	if(vecSize % nIter) vecSize++;

	nThread = (thread_t *)malloc(nIter * sizeof(thread_t));
	param = (struct param *)malloc(nIter * sizeof(struct param));

	getTime(&s);
	// Alloc & init input data
	ret = oclGlobalMalloc((void **)&a, nIter * vecSize * sizeof(float));
	assert(ret == gmacSuccess);
	valueInit(a, 1.0, nIter * vecSize);
	ret = oclGlobalMalloc((void **)&b, nIter * vecSize * sizeof(float));
	assert(ret == gmacSuccess);
	valueInit(b, 1.0, nIter * vecSize);

	// Alloc output data
	getTime(&t);
	printTime(&s, &t, "Alloc: ", "\n");

	for(n = 0; n < nIter; n++) {
		param[n].i = n;
		nThread[n] = thread_create(addVector, &(param[n]));
	}

	for(n = 0; n < nIter; n++) {
		thread_wait(nThread[n]);
	}

	oclFree(a);
	oclFree(b);

	float error = 0;
	for(n = 0; n < nIter; n++) {
		for(unsigned i = 0; i < vecSize; i++) {
			error += param[n].ptr[i] - 2;
		}
	}
	fprintf(stdout, "Total: %.02f\n", error);

	free(param);
	free(nThread);

    return error != 0;
}
