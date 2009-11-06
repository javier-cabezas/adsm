#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#include <pthread.h>

#include <gmac.h>

#include "utils.h"
#include "debug.h"

const unsigned rounds = 64;

const char *nIterStr = "GMAC_NITER";
const char *vecSizeStr = "GMAC_VECSIZE";

const unsigned nIterDefault = 1;
const size_t vecSizeDefault = 64 * 1024 * 1024;

unsigned nIter = 0;
size_t vecSize = 0;
const size_t blockSize = 512;

static float **s;

__global__ void vecAdd(float *c, float *a, float *b, size_t vecSize)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i >= vecSize) return;

	c[i] = a[i] + b[i];
}


void randInit(float *a, size_t vecSize)
{
	for(int i = 0; i < vecSize; i++) {
		a[i] = 1.0;
	}
}

void *addVector(void *ptr)
{
	float *a, *b;
	float **c = (float **)ptr;
	struct timeval s, t;
	gmacError_t ret = gmacSuccess;

	// Alloc & init input data
	ret = gmacMalloc((void **)&a, vecSize * sizeof(float));
	assert(ret == gmacSuccess);
	ret = gmacMalloc((void **)&b, vecSize * sizeof(float));
	assert(ret == gmacSuccess);

	gettimeofday(&s, NULL);
	for(int i = 0; i < rounds; i++) {
		randInit(a, vecSize);
		randInit(b, vecSize);
	}
	gettimeofday(&t, NULL);
	printAvgTime(&s, &t, "Alloc: ", "\n", rounds);

#if 0
	// Alloc output data
	ret = gmacMalloc((void **)c, vecSize * sizeof(float));
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
		//error += (a[i] - b[i]);
	}
	gettimeofday(&t, NULL);
	printTime(&s, &t, "Check: ", "\n");
	fprintf(stdout, "Error: %.02f\n", error);
#endif

	gmacFree(a);
	gmacFree(b);

	return NULL;
}


int main(int argc, char *argv[])
{
	pthread_t *nThread;
	unsigned n = 0;

	setParam<unsigned>(&nIter, nIterStr, nIterDefault);
	setParam<size_t>(&vecSize, vecSizeStr, vecSizeDefault);

	vecSize = vecSize / nIter;

	nThread = (pthread_t *)malloc(nIter * sizeof(pthread_t));
	s = (float **)malloc(nIter * sizeof(float **));

	srand(time(NULL));

	for(n = 0; n < nIter; n++) {
		pthread_create(&nThread[n], NULL, addVector, &s[n]);
	}

	for(n = 0; n < nIter; n++) {
		pthread_join(nThread[n], NULL);
	}

	float error = 0;
	for(n = 0; n < nIter; n++) {
		for(int i = 0; i < vecSize; i++) {
			error += s[n][i] - 2;
		}
	}
	fprintf(stdout, "Total: %.02f\n", error);

	free(s);
	free(nThread);

}
