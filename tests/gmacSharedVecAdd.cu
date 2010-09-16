#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#include <pthread.h>

#include <gmac.h>

#include "utils.h"
#include "debug.h"

const char *nIterStr = "GMAC_NITER";
const char *vecSizeStr = "GMAC_VECSIZE";

const unsigned nIterDefault = 4;
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
	struct timeval s, t;
	struct param *p = (struct param *)ptr;
	gmacError_t ret = gmacSuccess;

	ret = gmacMalloc((void **)&p->ptr, vecSize * sizeof(float));
	assert(ret == gmacSuccess);

	// Call the kernel
	dim3 Db(blockSize);
	dim3 Dg(vecSize / blockSize);
	if(vecSize % blockSize) Dg.x++;

	gettimeofday(&s, NULL);
	 vecAdd<<<Dg, Db>>>(gmacPtr((p->ptr)), gmacPtr(a + p->i * vecSize), gmacPtr(b + p->i * vecSize), vecSize);
	if(gmacThreadSynchronize() != gmacSuccess) CUFATAL();
	gettimeofday(&t, NULL);
	printTime(&s, &t, "Run: ", "\n");

	gettimeofday(&s, NULL);
	float error = 0;
	for(int i = 0; i < vecSize; i++) {
		error += p->ptr[i] - (a[i + p->i * vecSize] + b[i + p->i * vecSize]);
		//error += (a[i] - b[i]);
	}
	gettimeofday(&t, NULL);
	printTime(&s, &t, "Check: ", "\n");
	fprintf(stdout, "Error: %.02f\n", error);

    assert(error == 0);

	return NULL;
}


int main(int argc, char *argv[])
{
	pthread_t *nThread;
	unsigned n = 0;
	gmacError_t ret = gmacSuccess;
	struct timeval s, t;

	setParam<unsigned>(&nIter, nIterStr, nIterDefault);
	setParam<size_t>(&vecSize, vecSizeStr, vecSizeDefault);

	vecSize = vecSize / nIter;
	if(vecSize % nIter) vecSize++;

	nThread = (pthread_t *)malloc(nIter * sizeof(pthread_t));
	param = (struct param *)malloc(nIter * sizeof(struct param));

	gettimeofday(&s, NULL);
	// Alloc & init input data
	ret = gmacGlobalMalloc((void **)&a, nIter * vecSize * sizeof(float));
	assert(ret == gmacSuccess);
	valueInit(a, 1.0, nIter * vecSize);
	ret = gmacGlobalMalloc((void **)&b, nIter * vecSize * sizeof(float));
	assert(ret == gmacSuccess);
	valueInit(b, 1.0, nIter * vecSize);

	// Alloc output data
	gettimeofday(&t, NULL);
	printTime(&s, &t, "Alloc: ", "\n");

	for(n = 0; n < nIter; n++) {
		param[n].i = n;
		pthread_create(&nThread[n], NULL, addVector, &(param[n]));
	}

	for(n = 0; n < nIter; n++) {
		pthread_join(nThread[n], NULL);
	}

	gmacFree(a);
	gmacFree(b);

	float error = 0;
	for(n = 0; n < nIter; n++) {
		for(int i = 0; i < vecSize; i++) {
			error += param[n].ptr[i] - 2;
		}
	}
	fprintf(stdout, "Total: %.02f\n", error);

	free(param);
	free(nThread);

    return error != 0;
}
