#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#include <pthread.h>
#include <semaphore.h>

#include <gmac.h>

#include "utils.h"
#include "debug.h"

const char *nIterStr = "GMAC_NITER";
const char *vecSizeStr = "GMAC_VECSIZE";
const char *roundsStr = "GMAC_ROUNDS";

const unsigned nIterDefault = 4;
const size_t vecSizeDefault = 1024 * 1024;
const unsigned roundsDefault = 4;

unsigned nIter = 0;
size_t vecSize = 0;
unsigned rounds = 0;
const size_t blockSize = 512;

static pthread_t *nThread;
static int *ids;
static float **a;
static sem_t init;

__global__ void inc(float *a, float f, size_t size)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i >= size) return;

	a[i] += f;
}


void randInit(float *a, float f, size_t size)
{
	for(int i = 0; i < vecSize; i++) {
		a[i] = f;
	}
}

void *chain(void *ptr)
{
	int *id = (int *)ptr;
	gmacError_t ret = gmacSuccess;
	int n = 0, m = 0;

	ret = gmacMalloc((void **)&a[*id], vecSize * sizeof(float));
	assert(ret == gmacSuccess);
	randInit(a[*id], *id, vecSize);
	int next = (*id == nIter - 1) ? 0 : *id + 1;
	dim3 Db(blockSize);
	dim3 Dg(vecSize / blockSize);
	if(vecSize % blockSize) Dg.x++;

	sem_wait(&init);

	for(int i = 0; i < rounds; i++) {
		int current = *id - i;
		if(current < 0) current += nIter;
		//fprintf(stderr,"Thread %d(-> %d) +%d on %d\n", *id, next, *id, current);
		// Call the kernel
		inc<<<Dg, Db>>>(gmacPtr(a[current]), *id, vecSize);
		if(gmacThreadSynchronize() != gmacSuccess) CUFATAL();

		// Pass the context
		n++;
		gmacSendReceive(nThread[next]);
		m++;
	}
	int current = *id - rounds;
	if(current < 0) current += nIter;

	fprintf(stderr,"%d (Thread %d): %d sends\t%d receives\n", current, *id, n, m);
	float error = 0;
	for(int i = 0; i < vecSize; i++) {
		error += (a[current][i]);
	}
	fprintf(stderr,"%d (Thread %d): Error %f\n", current, *id, error / 1024);

	gmacFree(a[current]);

	return NULL;
}


int main(int argc, char *argv[])
{
	unsigned n = 0;

	setParam<unsigned>(&nIter, nIterStr, nIterDefault);
	setParam<size_t>(&vecSize, vecSizeStr, vecSizeDefault);
	setParam<unsigned>(&rounds, roundsStr, roundsDefault);
	sem_init(&init, 0, 0);

	nThread = (pthread_t *)malloc(nIter * sizeof(pthread_t));
	ids = (int *)malloc(nIter * sizeof(int));
	a = (float **)malloc(nIter * sizeof(float **));

	for(n = 0; n < nIter; n++) {
		ids[n] = n;
		pthread_create(&nThread[n], NULL, chain, &ids[n]);
	}

	for(n = 0; n < nIter; n++) sem_post(&init);
	
	for(n = 0; n < nIter; n++) {
		pthread_join(nThread[n], NULL);
	}


	free(ids);
	free(nThread);
}
