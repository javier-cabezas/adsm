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

const unsigned nIterDefault = 2;
const size_t vecSizeDefault = 1024 * 1024;

unsigned nIter = 0;
size_t vecSize = 0;
const size_t blockSize = 512;

static float **s;

__global__ void vecAdd(float *c, const float *a, const float *b, size_t vecSize)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i >= vecSize) return;

	c[i] = a[i] + b[i];
}

void *addVector(void *ptr)
{
	float *a, *b;
	float **c = (float **)ptr;
	struct timeval s, t;
	gmacError_t ret = gmacSuccess;

    gmacKernel_t kernel  = gmacKernel(vecAdd, float *, const float *, const float *, size_t);
	gettimeofday(&s, NULL);
	// Alloc & init input data
	ret = gmacMalloc((void **)&a, vecSize * sizeof(float));
	assert(ret == gmacSuccess);
	valueInit(a, 1.0, vecSize);
	ret = gmacMalloc((void **)&b, vecSize * sizeof(float));
	assert(ret == gmacSuccess);
	valueInit(b, 1.0, vecSize);

	// Alloc output data
	ret = gmacMalloc((void **)c, vecSize * sizeof(float));
	assert(ret == gmacSuccess);
	gettimeofday(&t, NULL);
	printTime(&s, &t, "Alloc: ", "\n");

    ret = gmacBind(a, kernel);
    assert(ret == gmacSuccess);
    ret = gmacBind(b, kernel);
    assert(ret == gmacSuccess);
    ret = gmacBind(*c, kernel);
    assert(ret == gmacSuccess);

	// Call the kernel
	dim3 Db(blockSize);
	dim3 Dg(vecSize / blockSize);
	if(vecSize % blockSize) Dg.x++;
	gettimeofday(&s, NULL);
	vecAdd<<<Dg, Db>>>(gmacPtr(*c), gmacPtr(a), gmacPtr(b), vecSize);
	if(gmacThreadSynchronize() != gmacSuccess) CUFATAL();
	gettimeofday(&t, NULL);
	printTime(&s, &t, "Run: ", "\n");

	gettimeofday(&s, NULL);
	float error = 0;
	for(int i = 0; i < vecSize; i++) {
		error += (*c)[i] - (a[i] + b[i]);
	}
	gettimeofday(&t, NULL);
	printTime(&s, &t, "Check: ", "\n");
	fprintf(stdout, "Error: %.02f\n", error);

	gmacFree(a);
	gmacFree(b);
	gmacFree(*c);

	return NULL;
}

int main(int argc, char *argv[])
{
	pthread_t *nThread;
	unsigned n = 0;
	struct timeval st, en;

	setParam<unsigned>(&nIter, nIterStr, nIterDefault);
	setParam<size_t>(&vecSize, vecSizeStr, vecSizeDefault);

	vecSize = vecSize / nIter;
	if(vecSize % nIter) vecSize++;

	nThread = (pthread_t *)malloc(nIter * sizeof(pthread_t));
	s = (float **)malloc(nIter * sizeof(float **));

	gettimeofday(&st, NULL);
	for(n = 0; n < nIter; n++) {
		pthread_create(&nThread[n], NULL, addVector, &s[n]);
	}

	for(n = 0; n < nIter; n++) {
		pthread_join(nThread[n], NULL);
	}

	gettimeofday(&en, NULL);
	printTime(&st, &en, "Total: ", "\n");

	free(s);
	free(nThread);
}
