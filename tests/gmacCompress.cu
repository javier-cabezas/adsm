#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#include <pthread.h>
#include <semaphore.h>

#include <gmac.h>

#include "utils.h"
#include "debug.h"

#include "gmacCompress.h"

const char *widthStr = "GMAC_WIDTH";
const char *heightStr = "GMAC_HEIGHT";
const char *framesStr = "GMAC_FRAMES";

const size_t widthDefault = 128;
const size_t heightDefault = 128;
const size_t framesDefault = 32;

size_t width = 0;
size_t height = 0;
size_t frames = 0;
const size_t blockSize = 16;

static float *quant_in, *idct_in;

static pthread_t dct_id, quant_id, idct_id;
static sem_t quant_data, idct_data;
static sem_t quant_free, idct_free;


void __randInit(float *a, size_t size)
{
	for(int i = 0; i < size; i++) {
		a[i] = 10.0 * rand() / RAND_MAX;
	}
}

void *dct_thread(void *args)
{
	float *in, *out;
	gmacError_t ret;

	ret = gmacMalloc((void **)&in, width * height * sizeof(float));
	assert(ret == gmacSuccess);
	ret = gmacMalloc((void **)&out, width * height * sizeof(float));
	assert(ret == gmacSuccess);

	dim3 Db(blockSize, blockSize);
	dim3 Dg(width / blockSize, height / blockSize);
	if(width % blockSize) Dg.x++;
	if(height % blockSize) Dg.y++;

	for(int i = 0; i < frames; i++) {
		__randInit(in, width * height);
		dct<<<Dg, Db>>>(gmacPtr(out), gmacPtr(in), width, height);
		ret = gmacThreadSynchronize();
		assert(ret == gmacSuccess);

		sem_wait(&quant_free); /* Wait for quant to use its data */
		gmacMemcpy(quant_in, out, width * height * sizeof(float));
		sem_post(&quant_data); /* Notify to Quant that data is ready */
	}

	gmacFree(in);
	gmacFree(out);

	return NULL;
}

void *quant_thread(void *args)
{
	float *out;
	gmacError_t ret;

	ret = gmacMalloc((void **)&quant_in, width * height * sizeof(float));
	assert(ret == gmacSuccess);
	ret = gmacMalloc((void **)&out, width * height * sizeof(float));
	assert(ret == gmacSuccess);

	dim3 Db(blockSize, blockSize);
	dim3 Dg(width / blockSize, height / blockSize);
	if(width % blockSize) Dg.x++;
	if(height % blockSize) Dg.y++;

	sem_post(&quant_free);

	for(int i = 0; i < frames; i++) {
		sem_wait(&quant_data);	/* Wait for data to be processed */
		quant<<<Dg, Db>>>(gmacPtr(quant_in), gmacPtr(out), width, height, 1e-6);
		ret = gmacThreadSynchronize();
		assert(ret == gmacSuccess);
		
		sem_wait(&idct_free); /* Wait for IDCT to use its data */
		gmacMemcpy(idct_in, out, width * height * sizeof(float));
		sem_post(&quant_free); /* Notify to DCT that Quant is waiting for data */
		sem_post(&idct_data); /* Nodify to IDCT that data is ready */
	}

	gmacFree(quant_in);
	gmacFree(out);

	return NULL;
}

void *idct_thread(void *args)
{
	float *out;
	gmacError_t ret;

	ret = gmacMalloc((void **)&idct_in, width * height * sizeof(float));
	assert(ret == gmacSuccess);
	ret = gmacMalloc((void **)&out, width * height * sizeof(float));
	assert(ret == gmacSuccess);

	dim3 Db(blockSize, blockSize);
	dim3 Dg(width / blockSize, height / blockSize);
	if(width % blockSize) Dg.x++;
	if(height % blockSize) Dg.y++;

	sem_post(&idct_free);

	for(int i = 0; i < frames; i++) {
		sem_wait(&idct_data);
		idct<<<Dg, Db>>>(gmacPtr(idct_in), gmacPtr(out), width, height);
		ret = gmacThreadSynchronize();
		assert(ret == gmacSuccess);

		sem_post(&idct_free);
	}


	gmacFree(idct_in);
	gmacFree(out);

	return NULL;
}


int main(int argc, char *argv[])
{
	struct timeval s,t;
	setParam<size_t>(&width, widthStr, widthDefault);
	setParam<size_t>(&height, heightStr, heightDefault);
	setParam<size_t>(&frames, framesStr, framesDefault);

	sem_init(&quant_data, 0, 0); 
	sem_init(&quant_free, 0, 0); 
	sem_init(&idct_data, 0, 0); 
	sem_init(&idct_free, 0, 0); 

	srand(time(NULL));

	gettimeofday(&s, NULL);

	pthread_create(&dct_id, NULL, dct_thread, NULL);
	pthread_create(&quant_id, NULL, quant_thread, NULL);
	pthread_create(&idct_id, NULL, idct_thread, NULL);

	pthread_join(dct_id, NULL);
	pthread_join(quant_id, NULL);
	pthread_join(idct_id, NULL);

	gettimeofday(&t, NULL);

	printTime(&s, &t, "Total: ", "\n");

}
