#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gmac.h>
#include <gmac/cuda.h>

#include "debug.h"

const size_t vecSize = 1024 * 1024;
const size_t blockSize = 512;

const char *msg = "Done!";

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


int main(int argc, char *argv[])
{
	float *a, *b, *c;

	srand(time(NULL));

	// Alloc & init input data
	if(gmacMalloc((void **)&a, vecSize * sizeof(float)) != gmacSuccess)
		CUFATAL();
	randInit(a, vecSize);
	if(gmacMalloc((void **)&b, vecSize * sizeof(float)) != gmacSuccess)
		CUFATAL();
	randInit(b, vecSize);

	// Alloc output data
	if(gmacMalloc((void **)&c, vecSize * sizeof(float)) != gmacSuccess)
		CUFATAL();

	// Call the kernel
	dim3 Db(blockSize);
	dim3 Dg(vecSize / blockSize);
	if(vecSize % blockSize) Db.x++;
	vecAdd<<<Dg, Db>>>(c, a, b);
	if(gmacThreadSynchronize() != gmacSuccess) CUFATAL();

	float error = 0;
	for(int i = 0; i < vecSize; i++) {
		error += c[i] - (a[i] + b[i]);
	}
	fprintf(stdout, "Error: %.02f\n", error);

	gmacFree(a);
	gmacFree(b);
	gmacFree(c);
}
