#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#include <pthread.h>
#include <semaphore.h>

#include <gmac.h>

#include "utils.h"
#include "debug.h"

#define PI 3.14159265358979

const char *widthStr = "GMAC_WIDTH";
const char *heightStr = "GMAC_HEIGHT";

const size_t widthDefault = 512;
const size_t heightDefault = 512;

size_t width = 0;
size_t height = 0;
const size_t blockSize = 16;

static pthread_t *nThread;
static int *ids;
//static float *intput, *temp, *output;
static sem_t init;

__shared__ float tile[blockSize][blockSize];

__global__ void dct(float *out, float *in, size_t width, size_t height)

{
	int l = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.x + blockIdx.x * blockDim.x;

	/* Pre-compute some values */
	float alpha, beta;
	if(k == 0) alpha = sqrtf(1.0 / width);
	else alpha = sqrtf(2.0 / width);
	if(l == 0) beta = sqrtf(1.0 / height);
	else beta = sqrtf(2.0 / height);

	float a = (PI / width) * k;
	float b = (PI / height) * l;

	float o = 0;
	for(int j = 0; j < height; j += blockDim.y) {
		for(int i = 0; i < width; i+= blockDim.x) {
			/* Calculate n and m values */
			int y = j + threadIdx.y;
			int x = i + threadIdx.x;

			/* Prefetch data in shared memory */
			if(x < width && y < height)
				tile[threadIdx.x][threadIdx.y] = in[y * width + x];
			__syncthreads();

			/* Compute the partial DCT */
			for(int m = 0; m < blockDim.y; m++) {
				for(int n = 0; n < blockDim.x; n++) {
					o += tile[m][n] * cosf(a * (n + i + 0.5)) * cosf(b * (m + j + 0.5));
				}
			}
			
			/* Done computing the DCT for the sub-block */
		}
	}

	if(k < width && l < height) {
		out[(l * width) + k] = alpha * beta * o;
	}
}

__global__ void idct(float *out, float *in, size_t width, size_t height)
{
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	/* Pre-compute some values */
	float alpha, beta;

	float a = (PI / width) * (x + 0.5);
	float b = (PI / height) * (y + 0.5);

	float o = 0;
				
	for(int j = 0; j < height; j += blockDim.y) {
		for(int i = 0; i < width; i+= blockDim.x) {
			/* Calculate n and m values */
			int l = j + threadIdx.y;
			int k = i + threadIdx.x;

			/* Prefetch data in shared memory */
			if(i + threadIdx.x < width && j + threadIdx.y < height)
				tile[threadIdx.x][threadIdx.y] = in[l * width + k];
			__syncthreads();

			/* Compute the partial IDCT */
			for(int m = 0; m < blockDim.y; m++) {
				for(int n = 0; n < blockDim.x; n++) {
					/* Pre-compute some values */
					if((n + i) == 0) alpha = sqrtf(1.0 / width);
					else alpha = sqrtf(2.0 / width);
					if((m + j) == 0) beta = sqrtf(1.0 / height);
					else beta = sqrtf(2.0 / height);
					o += alpha * beta * tile[m][n] * cosf(a * (n + i)) * cosf(b * (m + j));
				}
			}
			
			/* Done computing the DCT for the sub-block */
		}
	}

	if(x < width && y < height) {
		out[(y * width) + x] = o;
	}

}

__global__ void quant(float *out, float *in, size_t width, size_t height, float k)
{
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	if(x < width && y < height) {
		float f = fabsf(in[y * width + x]);
		if(f > k) out[(y * width) + x] = f;
		else out[(y * width) + x] = 0.0;
	}
}

void __randInit(float *a, size_t size)
{
	for(int i = 0; i < size; i++) {
		a[i] = 10.0 * rand() / RAND_MAX;
	}
}

void *chain(void *ptr)
{
	//int *id = (int *)ptr;
	float *a, *b, *c, *d;
	gmacError_t ret = gmacSuccess;

	ret = gmacMalloc((void **)&a, width * height * sizeof(float));
	assert(ret == gmacSuccess);
	ret = gmacMalloc((void **)&b, width * height * sizeof(float));
	assert(ret == gmacSuccess);
	ret = gmacMalloc((void **)&c, width * height * sizeof(float));
	assert(ret == gmacSuccess);
	ret = gmacMalloc((void **)&d, width * height * sizeof(float));
	assert(ret == gmacSuccess);


	__randInit(a, width * height);
	dim3 Db(blockSize, blockSize);
	dim3 Dg(width / blockSize, height / blockSize);
	if(width % blockSize) Dg.x++;
	if(height % blockSize) Dg.y++;


	sem_wait(&init);


	fprintf(stderr, "DCT\n");
	dct<<<Dg, Db>>>(gmacPtr(b), gmacPtr(a), width, height);
	ret = gmacThreadSynchronize();
	assert(ret == gmacSuccess);

	fprintf(stderr,"Quant\n");
	quant<<<Dg, Db>>>(gmacPtr(c), gmacPtr(b), width, height, 1e-6);
	ret = gmacThreadSynchronize();
	assert(ret == gmacSuccess);

	fprintf(stderr, "IDCT\n");
	idct<<<Dg, Db>>>(gmacPtr(d), gmacPtr(c), width, height);
	ret = gmacThreadSynchronize();
	assert(ret == gmacSuccess);

	float error = 0;
	for(int i = 0; i < width * height; i++)
		error += a[i] - d[i];
	fprintf(stderr,"Error %f\n", error);

	gmacFree(a);
	gmacFree(b);
	gmacFree(c);
	gmacFree(d);

	return NULL;
}


int main(int argc, char *argv[])
{
	unsigned n = 0;

	setParam<size_t>(&width, widthStr, widthDefault);
	setParam<size_t>(&height, heightStr, heightDefault);
	int nIter = 1;
	sem_init(&init, 0, 0);

	srand(time(NULL));

	nThread = (pthread_t *)malloc(nIter * sizeof(pthread_t));
	ids = (int *)malloc(nIter * sizeof(int));

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
