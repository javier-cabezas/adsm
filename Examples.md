# Simple (vector addition) #

## Kernel Code ##

Given this simple kernel that adds two vectors and writes the result into a third vector. This code is quite trivial, but it is enough to show how to program GPUs using GMAC.

```

__global__ void vecAdd(float *c, const float *a, const float *b, size_t size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= size) return;

    c[i] = a[i] + b[i];
}

```

## GMAC CPU Code ##

GMAC transparently handles the coherency of the data and uses a single pointer to refer to data structures that are used in the GPU.

```

#include <gmac/cuda.h>
#include <assert.h>

int main(int argc, char * argv[])
{
    float *a, *b, *c;

    /* 1- Allocate the input vectors */
    assert(gmacMalloc(&a, VECTOR_SIZE * sizeof(float)) == gmacSuccess);
    assert(gmacMalloc(&b, VECTOR_SIZE * sizeof(float)) == gmacSuccess);

    /* 2- Initialize the input vectors */
    for (int i = 0; i < VECTOR_SIZE; i++) {
        a[i] = 1.0f * rand();
        b[i] = 1.0f * rand();
    }

    /* 3- Allocate the output vector */
    assert(gmacMalloc(&c, VECTOR_SIZE * sizeof(float)) == gmacSuccess);

    /* 4- Invoke the kernel */
    dim3 block(BLOCK_SIZE);
    dim3 grid(VECTOR_SIZE / BLOCK_SIZE);
    if(VECTOR_SIZE % BLOCK_SIZE) grid.x++;

    vecAdd<<<grid, block>>>(c, a, b, VECTOR_SIZE);

    /* 5- Wait for kernel completion */
    assert(gmacThreadSynchronize() == gmacSuccess);v

    /* Check the result */
    for(int = 0; i < VECTOR_SIZE; i++) {
        assert(c[i] = a[i] + b[i]);
    }

    /* 6- Free shared structures */
    assert(gmacFree(a) == gmacSuccess);
    assert(gmacFree(b) == gmacSuccess);
    assert(gmacFree(c) == gmacSuccess);

    return 0;
}

```

## CUDA CPU Code ##

For comparison purposes, we show the analogous CUDA code the vector addition program would implement. CUDA makes the CPU code somewhat more complex, because two pointers are needed to refer to the input and output vectors. Moreover, the programmer has to explicitly request memory copies between CPU and GPU memories.


```

#include <cuda.h>
#include <assert.h>

int main(int argc, char * argv[])
{
    float *a, *b, *c;
    float *dev_a, *dev_b, *dev_c;

    /* 1- Allocate the input vectors in the host */
    assert((a = (float *) malloc(VECTOR_SIZE * sizeof(float))) != NULL);
    assert((b = (float *) malloc(VECTOR_SIZE * sizeof(float))) != NULL);

    /* 2- Allocate the input vectors in the device */
    assert(cudaMalloc(&dev_a, VECTOR_SIZE * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&dev_b, VECTOR_SIZE * sizeof(float)) == cudaSuccess);

    /* 3- Initialize the input vectors */
    for (int i = 0; i < VECTOR_SIZE; i++) {
        a[i] = 1.0f * rand();
        b[i] = 1.0f * rand();
    }

    /* 4- Transfer the input vectors to the device */
    assert(cudaMemcpy(dev_a, a, VECTOR_SIZE * sizeof(float),
           cudaMemcpyHostToDevice) == cudaSuccess)
    assert(cudaMemcpy(dev_b, b, VECTOR_SIZE * sizeof(float),
           cudaMemcpyHostToDevice) == cudaSuccess);

    /* 5- Allocate the output vector in the host */
    assert((c = (float *) malloc(VECTOR_SIZE * sizeof(float))) != NULL);

    /* 6- Allocate the output vector in the device */
    assert(cudaMalloc(&dev_c, VECTOR_SIZE * sizeof(float)) == cudaSuccess);

    /* 7- Invoke the kernel */
    dim3 block(BLOCK_SIZE);
    dim3 grid(VECTOR_SIZE / BLOCK_SIZE);
    if(VECTOR_SIZE % BLOCK_SIZE) grid.x++;

    vecAdd<<<grid, block>>>(dev_c, dev_a, dev_b, VECTOR_SIZE);

    /* 8- Copy the results to the host memory */
    assert(cudaMemcpy(c, dev_c, VECTOR_SIZE * sizeof(float),
           cudaMemcpyDeviceToHost) == cudaSuccess);

    /* 9- Wait for memory transfer */
    assert(cudaThreadSynchronize() == cudaSuccess);

    /* Check the result */
    for(int = 0; i < VECTOR_SIZE; i++) {
        assert(c[i] = a[i] + b[i]);
    }

    /* 10- Free host structures */
    free(a);
    free(b);
    free(c);

    /* 11- Free device structures */
    assert(cudaFree(dev_a) == cudaSuccess);
    assert(cudaFree(dev_b) == cudaSuccess);
    assert(cudaFree(dev_c) == cudaSuccess);

    return 0;
}

```