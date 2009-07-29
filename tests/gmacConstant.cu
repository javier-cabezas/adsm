#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#include <gmac.h>
#include <cuda.h>

#include "debug.h"


const int width = 16;
const int height = 16;

__constant__ float constant[width * height];

__global__ void vecAdd(float *c, int width, int height)
{
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

	c[y * width + x] = constant[y * width + x];
}


int main(int argc, char *argv[])
{
	float *data, *c;

	data = (float *)malloc(width * height * sizeof(float));
	for(int i = 0; i < width * height; i++)
		data[i] = i;
	assert(cudaMemcpyToSymbol(constant, data, width * height * sizeof(float)) == cudaSuccess);

	// Alloc output data
	if(gmacMalloc((void **)&c, width * height * sizeof(float)) != gmacSuccess)
		CUFATAL();

	// Call the kernel
	dim3 Db(width, height);
	dim3 Dg(1);
	vecAdd<<<Dg, Db>>>(c, width, height);
	if(gmacThreadSynchronize() != gmacSuccess) CUFATAL();

	for(int i = 0; i < width * height; i++) {
		if(c[i] == data[i]) continue;
		fprintf(stderr,"Error on %d (0x%x)\n", i, c[i]);
		abort();
	}
	fprintf(stderr,"Done!\n");

	gmacFree(c);
}
