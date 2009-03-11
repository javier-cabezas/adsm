#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <pthread.h>

#include <gmac.h>
#include <gmac/cuda.h>

#include "debug.h"

const size_t vecSize = 1024 * 1024;
const size_t blockSize = 512;
const unsigned nIter = 8;

__global__ void vecAdd(float *c, float *a, float *b)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i >= vecSize) return;

	c[i] = a[i] + b[i];
}


void randInit(float *a, size_t vecSize)
{
	for(int i = 0; i < vecSize; i++) {
		a[i] = rand() / (float)RAND_MAX;
	}
}

void *addVector(void *ptr)
{
	float *a, *b, *c;

	// Alloc & init input data
	if(gmacSafeMalloc((void **)&a, vecSize * sizeof(float)) != gmacSuccess)
		CUFATAL();
	randInit(a, vecSize);
	if(gmacSafeMalloc((void **)&b, vecSize * sizeof(float)) != gmacSuccess)
		CUFATAL();
	randInit(b, vecSize);

	// Alloc output data
	if(gmacSafeMalloc((void **)&c, vecSize * sizeof(float)) != gmacSuccess)
		CUFATAL();

	// Call the kernel
	dim3 Db(blockSize);
	dim3 Dg(vecSize / blockSize);
	if(vecSize % blockSize) Db.x++;
	vecAdd<<<Dg, Db>>>(gmacSafe(c), gmacSafe(a), gmacSafe(b));
	if(gmacThreadSynchronize() != gmacSuccess) CUFATAL();

	float error = 0;
	for(int i = 0; i < vecSize; i++) {
		error += c[i] - (a[i] + b[i]);
	}
	fprintf(stdout, "Error: %.02f\n", error);

	gmacFree(a);
	gmacFree(b);
	gmacFree(c);

	return NULL;
}


int main(int argc, char *argv[])
{
	pthread_t nThread[nIter];
	unsigned n = 0;

	srand(time(NULL));

	for(n = 0; n < nIter; n++) {
		pthread_create(&nThread[n], NULL, addVector, NULL);
	}

	for(n = 0; n < nIter; n++)
		pthread_join(nThread[n], NULL);

}
