#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#include <pthread.h>

#include <gmac.h>

#include "debug.h"

const char *nIterStr = "GMAC_NITER";
const char *vecSizeStr = "GMAC_VECSIZE";

const unsigned nIterDefault = 2;
const size_t vecSizeDefault = 1024 * 1024;

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

static inline void printTime(struct timeval *start, struct timeval *end, const char *pre, const char *post)
{
	double s, e;
	s = 1e6 * start->tv_sec + (start->tv_usec);
	e = 1e6 * end->tv_sec + (end->tv_usec);
	fprintf(stderr,"%s%f%s", pre, (e - s) / 1e6, post);
}

void *addVector(void *ptr)
{
	float *a, *b;
	float **c = (float **)ptr;
	struct timeval s, t;
	gmacError_t ret = gmacSuccess;

	gettimeofday(&s, NULL);
	// Alloc & init input data
	ret = gmacMalloc((void **)&a, vecSize * sizeof(float));
	assert(ret == gmacSuccess);
	randInit(a, vecSize);
	ret = gmacMalloc((void **)&b, vecSize * sizeof(float));
	assert(ret == gmacSuccess);
	randInit(b, vecSize);

	// Alloc output data
	ret = gmacMalloc((void **)c, vecSize * sizeof(float));
	assert(ret == gmacSuccess);
	gettimeofday(&t, NULL);
	printTime(&s, &t, "Alloc: ", "\n");

	// Call the kernel
	dim3 Db(blockSize);
	dim3 Dg(vecSize / blockSize);
	if(vecSize % blockSize) Db.x++;
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

	gmacFree(a);
	gmacFree(b);

	return NULL;
}

template<typename T>
void setParam(T *param, const char *str, const T def)
{
	const char *value = getenv(str);
	if(value != NULL) *param = atoi(value);
	if(*param == 0) *param = def;
}

int main(int argc, char *argv[])
{
	pthread_t *nThread;
	unsigned n = 0;

	setParam<unsigned>(&nIter, nIterStr, nIterDefault);
	setParam<size_t>(&vecSize, vecSizeStr, vecSizeDefault);

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
