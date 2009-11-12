#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gmac.h>

#include "utils.h"
#include "debug.h"


#define SIZE 1

const char *vecSizeStr = "GMAC_VECSIZE";
const size_t vecSizeDefault = 1024 * 1024;

size_t vecSize = 0;
const size_t blockSize = 512;

const char *msg = "Done!";

__global__ void vecAdd(float *c, float *a, float *b, size_t size)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i >= size) return;

	c[i] = a[i] + b[i];
}


int main(int argc, char *argv[])
{
	float *a = NULL, *b = NULL, *c = NULL;
	struct timeval s, t;

	setParam<size_t>(&vecSize, vecSizeStr, vecSizeDefault);
	fprintf(stdout, "Vector: %f\n", 1.0 * vecSize / 1024 / 1024);


	gettimeofday(&s, NULL);
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
	gettimeofday(&t, NULL);
	printTime(&s, &t, "Alloc: ", "\n");

	// Call the kernel
	gettimeofday(&s, NULL);
	dim3 Db(blockSize);
	dim3 Dg(vecSize / blockSize);
	if(vecSize % blockSize) Dg.x++;
	vecAdd<<<Dg, Db>>>(gmacPtr(c), gmacPtr(a), gmacPtr(b), vecSize);
	if(gmacThreadSynchronize() != gmacSuccess) CUFATAL();
	gettimeofday(&t, NULL);
	printTime(&s, &t, "Run: ", "\n");

	gettimeofday(&s, NULL);
	float error = 0;
	for(int i = 0; i < vecSize; i++) {
		error += c[i] - (a[i] + b[i]);
	}
	gettimeofday(&t, NULL);
	printTime(&s, &t, "Check: ", "\n");

	fprintf(stderr,"Error: %f\n", error);

	gmacFree(a);
	gmacFree(b);
	gmacFree(c);

}
