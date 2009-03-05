#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <pthread.h>
#include <semaphore.h>

#include <gmac.h>
#include <cuda.h>

#include "debug.h"

const size_t vecSize = 1024 * 1024;
const size_t blockSize = 512;
const unsigned nIter = 4;

const unsigned A = 0;
const unsigned B = 1;
float cpuBuffer[2][vecSize];

typedef struct {
	float *buffer;
	sem_t read;
	sem_t write;
} param_t;


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

void initParams(param_t *p, float *b)
{
	p->buffer = b;
	sem_init(&p->read, 0, 0);
	sem_init(&p->write, 0, 1);
}

void *produceData(void *ptr)
{
	param_t *param = (param_t *)ptr;
	unsigned n;
	for(n = 0; n < nIter; n++) {
		sem_wait(&param->write);
		randInit((float *)param->buffer, vecSize);
		sem_post(&param->read);
	}
	return NULL;
}

void *consumeData(void *ptr)
{	
	param_t *param = (param_t *)ptr;
	unsigned n;
	int i;
	float error = 0;

	for(n = 0; n < nIter; n++) {
		sem_wait(&param->read);
		for(i = 0; i < vecSize; i++) {
			error += param->buffer[i] - (cpuBuffer[A][i] + cpuBuffer[B][i]);
		}
		fprintf(stdout, "Error (%d): %.02f\n", n, error);
		sem_post(&param->write);
	}
	return NULL;
}

int main(int argc, char *argv[])
{
	float *a, *b, *c;
	param_t aParam, bParam, cParam;
	pthread_t aThread, bThread, cThread;
	unsigned n = 0;

	srand(time(NULL));

	// Alloc & init input data
	if(cudaMalloc((void **)&a, vecSize * sizeof(float)) != cudaSuccess)
		CUFATAL();
	initParams(&aParam, a);
	if(cudaMalloc((void **)&b, vecSize * sizeof(float)) != cudaSuccess)
		CUFATAL();
	initParams(&bParam, b);
	// Alloc output data
	if(cudaMalloc((void **)&c, vecSize * sizeof(float)) != cudaSuccess)
		CUFATAL();
	initParams(&cParam, c);


	// Create consumer-producer threads
	pthread_create(&aThread, NULL, produceData, (void *)&aParam);
	pthread_create(&bThread, NULL, produceData, (void *)&bParam);
	pthread_create(&cThread, NULL, consumeData, (void *)&cParam);


	// Call the kernel
	dim3 Db(blockSize);
	dim3 Dg(vecSize / blockSize);
	if(vecSize % blockSize) Db.x++;

	for(n = 0; n < nIter; n++) {
		// Wait for the producers to write the input data
		sem_wait(&aParam.read);
		sem_wait(&bParam.read);
		// Wait for the consumer to read the output data
		sem_wait(&cParam.write);
		// Save the current input data
		memcpy(cpuBuffer[A], a, vecSize * sizeof(float));
		memcpy(cpuBuffer[B], b, vecSize * sizeof(float));
		// Launch the kernel
		vecAdd<<<Dg, Db>>>(c, a, b);
		// Let the produces to write new input data
		sem_post(&aParam.write);
		sem_post(&bParam.write);
		if(cudaThreadSynchronize() != cudaSuccess) CUFATAL();
		// Let the consumer to read the output data
		sem_post(&cParam.read);
	}

	pthread_join(aThread, NULL);
	pthread_join(bThread, NULL);
	pthread_join(cThread, NULL);

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
}
