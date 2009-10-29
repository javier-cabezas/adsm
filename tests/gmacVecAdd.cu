#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gmac.h>

#include "utils.h"
#include "debug.h"


#define SIZE 1

const size_t vecSize = 4 * 1024 * 1024;
const size_t blockSize = 512;

const char *msg = "Done!";

__global__ void vecAdd(float *c, float *a, float *b, size_t size)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i >= size) return;

	c[i] = a[i] + b[i];
}


void randInit(float *a, size_t size)
{
	for(int i = 0; i < size; i++) {
		a[i] = 1.0 * rand();
	}
}

int main(int argc, char *argv[])
{
	float *a = NULL, *b = NULL, *c = NULL;
	struct timeval s, t;
	size_t size = 0;

	if(argv[SIZE] != NULL) size = atoi(argv[SIZE]);
	if(size == 0) size = vecSize;

	srand(time(NULL));

	gettimeofday(&s, NULL);
	// Alloc & init input data
	if(gmacMalloc((void **)&a, size * sizeof(float)) != gmacSuccess)
		CUFATAL();
	randInit(a, size);
	if(gmacMalloc((void **)&b, size * sizeof(float)) != gmacSuccess)
		CUFATAL();
	randInit(b, size);
	// Alloc output data
	if(gmacMalloc((void **)&c, size * sizeof(float)) != gmacSuccess)
		CUFATAL();
	gettimeofday(&t, NULL);
	printTime(&s, &t, "Alloc: ", "\n");

	// Call the kernel
	dim3 Db(blockSize);
	dim3 Dg(size / blockSize);
	if(size % blockSize) Dg.x++;
	gettimeofday(&s, NULL);
	vecAdd<<<Dg, Db>>>(gmacPtr(c), gmacPtr(a), gmacPtr(b), size);
	if(gmacThreadSynchronize() != gmacSuccess) CUFATAL();
	gettimeofday(&t, NULL);
	printTime(&s, &t, "Run: ", "\n");

	gettimeofday(&s, NULL);
	float error = 0;
	for(int i = 0; i < size; i++) {
		error += c[i] - (a[i] + b[i]);
	}
	gettimeofday(&t, NULL);
	printTime(&s, &t, "Check: ", "\n");

	fprintf(stderr,"Error: %f\n", error);

	gmacFree(a);
	gmacFree(b);
	gmacFree(c);

}
