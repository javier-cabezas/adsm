#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gmac.h>

#include "utils.h"
#include "debug.h"

const char *nIterStr = "GMAC_NITER";
const char *vecSizeStr = "GMAC_VECSIZE";

const unsigned nIterDefault = 1;
const size_t vecSizeDefault = 1024 * 1024;

unsigned nIter = 0;
size_t vecSize = 0;
const size_t blockSize = 512;


static float *a, *b;
static struct param {
	int i;
	float *ptr;
} *param;


__global__ void vecAdd(float *c, float *a, float *b, size_t vecSize)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i >= vecSize) return;

	c[i] = a[i] + b[i];
}

void *addVector(void *ptr)
{
	gmactime_t s, t;
	struct param *p = (struct param *)ptr;
	gmacError_t ret = gmacSuccess;

	ret = gmacMalloc((void **)&p->ptr, vecSize * sizeof(float));
	assert(ret == gmacSuccess);

	// Call the kernel
	dim3 Db(blockSize);
	dim3 Dg(int(vecSize / blockSize));
	if(vecSize % blockSize) Dg.x++;

	getTime(&s);
	 vecAdd<<<Dg, Db>>>(gmacPtr((p->ptr)), gmacPtr(a + p->i * vecSize), gmacPtr(b + p->i * vecSize), vecSize);
	if(gmacThreadSynchronize() != gmacSuccess) CUFATAL();
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

    assert(error == 0);

	return NULL;
}


int main(int argc, char *argv[])
{
	thread_t *nThread;
	unsigned n = 0;
	gmacError_t ret = gmacSuccess;
	gmactime_t s, t;

	setParam<unsigned>(&nIter, nIterStr, nIterDefault);
	setParam<size_t>(&vecSize, vecSizeStr, vecSizeDefault);

	vecSize = vecSize / nIter;
	if(vecSize % nIter) vecSize++;

	nThread = (thread_t *)malloc(nIter * sizeof(thread_t));
	param = (struct param *)malloc(nIter * sizeof(struct param));

	getTime(&s);
	// Alloc & init input data
	ret = gmacGlobalMalloc((void **)&a, nIter * vecSize * sizeof(float));
	assert(ret == gmacSuccess);
	valueInit(a, 1.0, nIter * vecSize);
	ret = gmacGlobalMalloc((void **)&b, nIter * vecSize * sizeof(float));
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

	gmacFree(a);
	gmacFree(b);

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
