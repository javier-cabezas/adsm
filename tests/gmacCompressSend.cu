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

static float *dct_in, *quant_in, *idct_in = NULL;
static float *dct_out, *quant_out, *idct_out = NULL;

static pthread_t dct_id, quant_id, idct_id;
static sem_t init;


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

	sem_wait(&init);

	for(int i = 0; i < frames; i++) {
		fprintf(stderr,"Frame %d starts\n", i);
		__randInit(in, width * height);
		dct<<<Dg, Db>>>(gmacPtr(out), gmacPtr(in), width, height);
		ret = gmacThreadSynchronize();
		assert(ret == gmacSuccess);

		quant_in = out;
		quant_out = in;
		gmacSendReceive(quant_id);
		in = dct_in;
		out = dct_out;
	}

	// Move two stages the pipeline
	gmacSendReceive(quant_id);
	gmacSendReceive(quant_id);

	gmacFree(in);
	gmacFree(out);

	return NULL;
}

void *quant_thread(void *args)
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

	sem_wait(&init);

	gmacSendReceive(idct_id);
	in = quant_in;
	out = quant_out;

	for(int i = 0; i < frames; i++) {
		quant<<<Dg, Db>>>(gmacPtr(out), gmacPtr(in), width, height, 1e-6);
		ret = gmacThreadSynchronize();
		assert(ret == gmacSuccess);
		
		idct_in = out;
		idct_out = in;
		gmacSendReceive(idct_id);
		in = quant_in;
		out = quant_out;
	}

	// Move one stage the pipeline stages the pipeline
	gmacSendReceive(idct_id);

	gmacFree(in);
	gmacFree(out);

	return NULL;
}

void *idct_thread(void *args)
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

	sem_wait(&init);

	while(idct_in == NULL) {
		dct_in = out;
		dct_out = in;
		gmacSendReceive(dct_id);
	}

	in = idct_in;
	out = idct_out;


	for(int i = 0; i < frames; i++) {
		idct<<<Dg, Db>>>(gmacPtr(out), gmacPtr(in), width, height);
		ret = gmacThreadSynchronize();
		assert(ret == gmacSuccess);

		fprintf(stderr,"Frame %d done!\n", i);
		dct_in = out;
		dct_out = in;
		gmacSendReceive(dct_id);
		in = idct_in;
		out = idct_out;
	}

	gmacFree(in);
	gmacFree(out);

	return NULL;
}


int main(int argc, char *argv[])
{
	setParam<size_t>(&width, widthStr, widthDefault);
	setParam<size_t>(&height, heightStr, heightDefault);
	setParam<size_t>(&frames, framesStr, framesDefault);

	sem_init(&init, 0, 0);

	srand(time(NULL));


	pthread_create(&dct_id, NULL, dct_thread, NULL);
	pthread_create(&quant_id, NULL, quant_thread, NULL);
	pthread_create(&idct_id, NULL, idct_thread, NULL);

	for(int i = 0; i < 3; i++) sem_post(&init);

	pthread_join(dct_id, NULL);
	pthread_join(quant_id, NULL);
	pthread_join(idct_id, NULL);
}
