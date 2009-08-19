#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gmac.h>

#include <sys/time.h>

#include "debug.h"

size_t vecSize = 1 * 1024 * 1024;
const size_t blockSize = 512;

const char *msg = "Done!";

__global__ void vecAdd(float *c, float *a, float *b, size_t vecSize)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i >= vecSize) return;

	c[i] = a[i] + b[i];
}


void randInit(float *a, size_t vecSize)
{
	for(int i = 0; i < vecSize; i++) {
		a[i] = 1.0 * i;
		//a[i] = rand() / (float)RAND_MAX;
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
	float *a, *b, *c;
	struct timeval s, t;

	const char *vecStr = getenv("VECTORSIZE");
	if(vecStr != NULL) vecSize = atoi(vecStr) * 1024 * 1024;
	fprintf(stderr,"Vector %dMB\n", vecSize);
	srand(time(NULL));
	// Alloc & init input data
	if(gmacMalloc((void **)&a, vecSize * sizeof(float)) != gmacSuccess)
		CUFATAL();
	if(gmacMalloc((void **)&b, vecSize * sizeof(float)) != gmacSuccess)
		CUFATAL();
	// Alloc output data
	if(gmacMalloc((void **)&c, vecSize * sizeof(float)) != gmacSuccess)
		CUFATAL();

	gettimeofday(&s, NULL);
	randInit(a, vecSize);
	randInit(b, vecSize);

	// Call the kernel
	dim3 Db(blockSize);
	dim3 Dg(vecSize / blockSize);
	if(vecSize % blockSize) Db.x++;
	vecAdd<<<Dg, Db>>>(c, a, b, vecSize);
	gettimeofday(&t, NULL);
	printTime(&s, &t, " ");

	if(gmacThreadSynchronize() != gmacSuccess) CUFATAL();


	gettimeofday(&s, NULL);
	float error = 0;
	for(int i = 0; i < vecSize; i++) {
		error += c[i] - (a[i] + b[i]);
	}
	gettimeofday(&t, NULL);
	printTime(&s, &t, "\n");

	fprintf(stderr,"Error: %f\n", error);

	gmacFree(a);
	gmacFree(b);
	gmacFree(c);

}
