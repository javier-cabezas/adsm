#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gmac.h>

#include <sys/time.h>

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


static inline void printTime(struct timeval *start, struct timeval *end, const char *str)
{
	double s, e;
	s = 1e6 * start->tv_sec + (start->tv_usec);
	e = 1e6 * end->tv_sec + (end->tv_usec);
	fprintf(stdout,"%f%s", (e - s) / 1e6, str);
}

int main(int argc, char *argv[])
{
	float *a = NULL, *b = NULL, *c = NULL;
	struct timeval s, t;
	size_t size = 0;

	if(argv[SIZE] != NULL) size = atoi(argv[SIZE]);
	if(size == 0) size = vecSize;

	fprintf(stderr,"Vector %dMB\n", size);
	srand(time(NULL));
	// Alloc & init input data
	if(gmacMalloc((void **)&a, size * sizeof(float)) != gmacSuccess)
		CUFATAL();
	if(gmacMalloc((void **)&b, size * sizeof(float)) != gmacSuccess)
		CUFATAL();
	// Alloc output data
	if(gmacMalloc((void **)&c, size * sizeof(float)) != gmacSuccess)
		CUFATAL();

	gettimeofday(&s, NULL);
	randInit(a, size);
	randInit(b, size);

	// Call the kernel
	dim3 Db(blockSize);
	dim3 Dg(size / blockSize);
	if(size % blockSize) Db.x++;
	vecAdd<<<Dg, Db>>>(gmacPtr(c), gmacPtr(a), gmacPtr(b), size);
	gettimeofday(&t, NULL);
	printTime(&s, &t, " ");

	gettimeofday(&s, NULL);
	if(gmacThreadSynchronize() != gmacSuccess) CUFATAL();
	gettimeofday(&t, NULL);
	printTime(&s, &t, " ");


	gettimeofday(&s, NULL);
	float error = 0;
	for(int i = 0; i < size; i++) {
		error += c[i] - (a[i] + b[i]);
	}
	gettimeofday(&t, NULL);
	printTime(&s, &t, "\n");

	fprintf(stderr,"Error: %f\n", error);

	gmacFree(a);
	gmacFree(b);
	gmacFree(c);

}
