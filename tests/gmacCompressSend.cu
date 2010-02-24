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


typedef struct stage {
	pthread_t id;
	sem_t free;
	float *in;
	float *out;
	float *next_in;
	float *next_out;
} stage_t;

stage_t s_dct, s_quant, s_idct;


void __randInit(float *a, size_t size)
{
	for(int i = 0; i < size; i++) {
		a[i] = 10.0 * rand() / RAND_MAX;
	}
}

void nextStage(stage_t *current, stage_t *next)
{
	if(next != NULL) {
		sem_wait(&next->free);
		next->next_in = current->out;
		next->next_out = current->in;
		gmacSendReceive(next->id);
	}
	if(current != NULL) {
		current->in = current->next_in;
		current->out = current->next_out;
		sem_post(&current->free);
	}
}

pthread_barrier_t barrierInit;

void *dct_thread(void *args)
{
	gmacError_t ret;

    pthread_barrier_wait(&barrierInit);

	dim3 Db(blockSize, blockSize);
	dim3 Dg(width / blockSize, height / blockSize);
	if(width % blockSize) Dg.x++;
	if(height % blockSize) Dg.y++;

	//sem_post(&s_dct.free);

	for(int i = 0; i < frames; i++) {
		//fprintf(stderr,"DCT %d init\n", i);
		ret = gmacMalloc((void **)&s_dct.in, width * height * sizeof(float));
		assert(ret == gmacSuccess);
		ret = gmacMalloc((void **)&s_dct.out, width * height * sizeof(float));
		assert(ret == gmacSuccess);

		__randInit(s_dct.in, width * height);
		dct<<<Dg, Db>>>(gmacPtr(s_dct.out), gmacPtr(s_dct.in), width, height);
		ret = gmacThreadSynchronize();
		assert(ret == gmacSuccess);

		//fprintf(stderr,"DCT %d done\n", i);
		
		sem_wait(&s_quant.free);
		s_quant.next_in = s_dct.out;
		s_quant.next_out = s_dct.in;
		gmacSendReceive(s_quant.id);
		//nextStage(NULL, &s_quant);	

		//assert(gmacFree(&s_dct.in) == gmacSuccess);
		//assert(gmacFree(&s_dct.out) == gmacSuccess);
	}

	ret = gmacMalloc((void **)&s_dct.in, width * height * sizeof(float));
	assert(ret == gmacSuccess);
	ret = gmacMalloc((void **)&s_dct.out, width * height * sizeof(float));
	assert(ret == gmacSuccess);

	sem_wait(&s_quant.free);
	s_quant.next_in = s_dct.out;
	s_quant.next_out = s_dct.in;
	gmacSendReceive(s_quant.id);
//	nextStage(NULL, &s_quant);

	ret = gmacMalloc((void **)&s_dct.in, width * height * sizeof(float));
	assert(ret == gmacSuccess);
	ret = gmacMalloc((void **)&s_dct.out, width * height * sizeof(float));
	assert(ret == gmacSuccess);

	sem_wait(&s_quant.free);
	s_quant.next_in = s_dct.out;
	s_quant.next_out = s_dct.in;
	gmacSendReceive(s_quant.id);
//	nextStage(NULL, &s_quant);

//	gmacFree(s_dct.in);
//	gmacFree(s_dct.out);

	return NULL;
}

void *quant_thread(void *args)
{
	gmacError_t ret;

    pthread_barrier_wait(&barrierInit);

	//ret = gmacMalloc((void **)&s_quant.in, width * height * sizeof(float));
	//assert(ret == gmacSuccess);
	//ret = gmacMalloc((void **)&s_quant.out, width * height * sizeof(float));
	//assert(ret == gmacSuccess);

	dim3 Db(blockSize, blockSize);
	dim3 Dg(width / blockSize, height / blockSize);
	if(width % blockSize) Dg.x++;
	if(height % blockSize) Dg.y++;

	sem_post(&s_quant.free);
	nextStage(&s_quant, &s_idct);

	for(int i = 0; i < frames; i++) {
		//fprintf(stderr,"Quant %d init\n", i);
		quant<<<Dg, Db>>>(gmacPtr(s_quant.out), gmacPtr(s_quant.in), width, height, 1e-6);
		ret = gmacThreadSynchronize();
		assert(ret == gmacSuccess);
		
		//fprintf(stderr,"Quant %d done\n", i);
		nextStage(&s_quant, &s_idct);
	}

	// Move one stage the pipeline stages the pipeline
	nextStage(&s_quant, &s_idct);

	//gmacFree(s_quant.in);
	//gmacFree(s_quant.out);

	return NULL;
}

void *idct_thread(void *args)
{
	gmacError_t ret;

    pthread_barrier_wait(&barrierInit);

	//ret = gmacMalloc((void **)&s_idct.in, width * height * sizeof(float));
	//assert(ret == gmacSuccess);
	//ret = gmacMalloc((void **)&s_idct.out, width * height * sizeof(float));
	//assert(ret == gmacSuccess);

	dim3 Db(blockSize, blockSize);
	dim3 Dg(width / blockSize, height / blockSize);
	if(width % blockSize) Dg.x++;
	if(height % blockSize) Dg.y++;

	sem_post(&s_idct.free);
	gmacSendReceive(s_dct.id);
	nextStage(&s_idct, NULL);
	gmacSendReceive(s_dct.id);
	nextStage(&s_idct, NULL);
	//assert(gmacFree(s_idct.in) == gmacSuccess);
	//assert(gmacFree(s_idct.out) == gmacSuccess);

	for(int i = 0; i < frames; i++) {
		//fprintf(stderr,"IDCT %d init\n", i);
		idct<<<Dg, Db>>>(gmacPtr(s_idct.out), gmacPtr(s_idct.in), width, height);
		ret = gmacThreadSynchronize();
		assert(ret == gmacSuccess);

		//fprintf(stderr,"IDCT %d done\n", i);

		assert(gmacFree(s_idct.in) == gmacSuccess);
		assert(gmacFree(s_idct.out) == gmacSuccess);

		gmacSendReceive(s_dct.id);
		nextStage(&s_idct, NULL);
	}

	gmacFree(s_idct.in);
	gmacFree(s_idct.out);

	return NULL;
}


int main(int argc, char *argv[])
{
	struct timeval s,t;
	setParam<size_t>(&width, widthStr, widthDefault);
	setParam<size_t>(&height, heightStr, heightDefault);
	setParam<size_t>(&frames, framesStr, framesDefault);

	//sem_init(&s_dct.free, 0, 0);
	sem_init(&s_quant.free, 0, 0);
	sem_init(&s_idct.free, 0, 0);

	srand(time(NULL));

	gettimeofday(&s, NULL);

    pthread_barrier_init(&barrierInit, NULL, 3);

	pthread_create(&s_dct.id, NULL, dct_thread, NULL);
	pthread_create(&s_quant.id, NULL, quant_thread, NULL);
	pthread_create(&s_idct.id, NULL, idct_thread, NULL);

	pthread_join(s_dct.id, NULL);
	pthread_join(s_quant.id, NULL);
	pthread_join(s_idct.id, NULL);

	gettimeofday(&t, NULL);

	printTime(&s, &t, "Total: ", "\n");

    return 0;
}
