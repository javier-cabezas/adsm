#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#include <gmac/cuda.h>
#include <cuda.h>

#include "debug.h"


const unsigned width = 16;
const unsigned height = 16;

__constant__ float constant[width * height];

__global__ void vecAdd(float *c, unsigned width, unsigned height)
{
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

	c[y * width + x] = constant[y * width + x];
}


int main(int argc, char *argv[])
{
	float *data, *c;

	data = (float *)malloc(width * height * sizeof(float));
	for(unsigned i = 0; i < width * height; i++)
		data[i] = float(i);
	assert(cudaMemcpyToSymbol(constant, data, width * height * sizeof(float)) == cudaSuccess);

	// Alloc output data
	if(gmacMalloc((void **)&c, width * height * sizeof(float)) != gmacSuccess)
		CUFATAL();

	// Call the kernel
	dim3 Db(width, height);
	dim3 Dg(1);
	vecAdd<<<Dg, Db>>>(gmacPtr(c), width, height);
	if(gmacThreadSynchronize() != gmacSuccess) CUFATAL();

	for(unsigned i = 0; i < width * height; i++) {
		if(c[i] == data[i]) continue;
		fprintf(stderr,"Error on %d (%f)\n", i, c[i]);
		abort();
	}
	fprintf(stderr,"Done!\n");

	gmacFree(c);

    return 0;
}
