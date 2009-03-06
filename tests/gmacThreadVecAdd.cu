#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <pthread.h>

#include <gmac.h>
#include <gmac/cuda.h>

#include "debug.h"

const size_t vecSize = 1024 * 1024;
const size_t blockSize = 512;
const unsigned nIter = 4;

__global__ void vecAdd(float *c, float *a, float *b)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i >= vecSize) return;

	c[i] = a[i] + b[i];
}


void randInit(float *a, size_t vecSize)
{
	for(int i = 0; i < vecSize; i++)
		a[i] = rand() / (float)RAND_MAX;
}

void *addVector(void *ptr)
{
	float *a, *b, *c;

	// Alloc & init input data
	if(cudaMalloc((void **)&a, vecSize * sizeof(float)) != cudaSuccess)
		CUFATAL();
	randInit(a, vecSize);
	if(cudaMalloc((void **)&b, vecSize * sizeof(float)) != cudaSuccess)
		CUFATAL();
	randInit(b, vecSize);

	// Alloc output data
	if(cudaMalloc((void **)&c, vecSize * sizeof(float)) != cudaSuccess)
		CUFATAL();

	// Call the kernel
	dim3 Db(blockSize);
	dim3 Dg(vecSize / blockSize);
	if(vecSize % blockSize) Db.x++;
	vecAdd<<<Dg, Db>>>(c, a, b);
	if(cudaThreadSynchronize() != cudaSuccess) CUFATAL();

	float error = 0;
	for(int i = 0; i < vecSize; i++) {
		error += c[i] - (a[i] + b[i]);
	}
	fprintf(stdout, "Error: %.02f\n", error);

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);

	return NULL;
}


int main(int argc, char *argv[])
{
	pthread_t nThread[nIter];
	unsigned n = 0;

	srand(time(NULL));

	for(n = 0; n < nIter; n++)
		pthread_create(&nThread[n], NULL, addVector, NULL);

	for(n = 0; n < nIter; n++)
		pthread_join(nThread[n], NULL);
}
