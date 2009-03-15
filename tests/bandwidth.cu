#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>

#include <cuda.h>

#include "debug.h"

const size_t buff_size = 512 * 1024 * 1024;
const size_t step_size = 4 * 1024;
const int iters = 64;
const size_t block_size = 512;

static uint8_t *cpu, *dev;

typedef unsigned long long usec_t;
typedef struct {
	double in;
	double out;
} stamp_t;

__global__ void null(uint8_t *p)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i > buff_size) return;
	p[i] = 0;
}

static inline usec_t get_time()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	usec_t tm = tv.tv_usec + 1000000 * tv.tv_sec;
	return tm;
}

static void kernel(int s)
{
	dim3 Db(block_size);
	dim3 Dg(s / block_size);
	if(s % block_size) Db.x++;
	null<<<Dg, Db>>>(dev);
	if(cudaThreadSynchronize() != cudaSuccess) CUFATAL();

}

void transfer(stamp_t *stamp, int s)
{
	usec_t start, end;
	int i;

	stamp->in = stamp->out = 0;
	for(i = 0; i < iters; i++) {
		start = get_time();
		if(cudaMemcpy(dev, cpu, s, cudaMemcpyHostToDevice) != cudaSuccess)
			CUFATAL();
		end = get_time();
		stamp->in += end - start;
		kernel(s);

		start = get_time();
		if(cudaMemcpy(cpu, dev, s, cudaMemcpyDeviceToHost) != cudaSuccess)
			CUFATAL();
		end = get_time();
		stamp->out += end - start;
	}
	stamp->in = stamp->in / iters;
	stamp->out = stamp->out /iters;
}


int main(int argc, char *argv[])
{
	stamp_t stamp;
	int i;

	// Alloc & init input data
	if((cpu = (uint8_t *)malloc(buff_size)) == NULL)
		FATAL();
	if(cudaMalloc((void **)&dev, buff_size) != cudaSuccess)
		CUFATAL();

	// Transfer data
	fprintf(stdout, "Bytes\tIn Time\tIn Bandwidth\tOut Time\tOut Bandwidth\n");
	for(i = step_size; i < buff_size; i += i) {
		transfer(&stamp, i);
		fprintf(stdout, "%d\t%f\t%f\t%f\t%f\n", i,
			stamp.in, i * 8.0 / stamp.in / 1000.0,
			stamp.out, i * 8.0 / stamp.out / 1000.0);
	}

	// Release memory
	cudaFree(dev);
	free(cpu);
}
