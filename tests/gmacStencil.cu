#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

#include <gmac.h>

#include "utils.h"
#include "debug.h"

#define ITERATIONS 1000

#define ARGS    1
#define STENCIL 4

#define INIT_VAL 200
#define VELOCITY 2000

static size_t dimRealElems = 256;
static size_t dimElems     = dimRealElems + 2 * STENCIL;

#define ELEMS_3D(d) (d * d * d)
#define SIZE_3D(d)  (ELEMS_3D(d) * sizeof(float))

#define ELEMS_2D(d) (d * d)
#define SIZE_2D(d)  (ELEMS_2D(d) * sizeof(float))

__constant__
float devC00;
__constant__
float devZ1;
__constant__
float devZ2;
__constant__
float devZ3;
__constant__
float devZ4;
__constant__
float devX1;
__constant__
float devX2;
__constant__
float devX3;
__constant__
float devX4;
__constant__
float devY1;
__constant__
float devY2;
__constant__
float devY3;
__constant__
float devY4;


template <uint32_t STENCIL_TILE_XSIZE, uint32_t STENCIL_TILE_YSIZE>
__global__
void
kernelStencil(const float * u1,
              const float * u2,
              float * u3,
              const float * v,
              const float dt2,
              const uint32_t dimZ,
              const uint32_t dimRealZ,
              const uint32_t dimZX,
              const uint32_t dimRealZX,
              const int slices)
{
    __shared__
        __align__(0x10)
        float s_data[(STENCIL_TILE_YSIZE + 2 * STENCIL) * (STENCIL_TILE_XSIZE + 2 * STENCIL)];

    uint32_t ix = (blockIdx.x * blockDim.x + threadIdx.x);
    uint32_t iy = (blockIdx.y * blockDim.y + threadIdx.y);

    uint32_t tx = threadIdx.x + STENCIL; // thread’s x-index into corresponding shared memory tile (adjusted for halos)
    uint32_t ty = threadIdx.y + STENCIL; // thread’s y-index into corresponding shared memory tile (adjusted for halos)

    int32_t index     = (iy * dimZ     + ix) + 3 * dimZX; // index for reading input
    int32_t realIndex = (iy * dimRealZ + ix); // index for reading/writing from/to structures without ghost area

#define TILE_OFFSET_LINE (STENCIL_TILE_XSIZE + 2 * STENCIL)
#define SH(x,y) ((y) * TILE_OFFSET_LINE + x)
#define SH_X(off) (uint32_t(int32_t(SH(tx, ty)) + (off)))
#define SH_Y(off) (uint32_t(int32_t(SH(tx, ty)) + int32_t(off) * int32_t(TILE_OFFSET_LINE)))

    float4 front;
    float4 back;

    float current;

    // fill the "in-front" and "behind" data
    back.z  = u2[index - 3 * dimZX];
    back.y  = u2[index - 2 * dimZX];
    back.x  = u2[index - 1 * dimZX];
    current = u2[index            ];
    front.w = u2[index + 1 * dimZX];
    front.z = u2[index + 2 * dimZX];
    front.y = u2[index + 3 * dimZX];
    front.x = u2[index + 4 * dimZX];

    //int signY = (threadIdx.y - STENCIL) >> 31;
    for (int k = 0; k < (slices - 2 * STENCIL); k++) {
        float tmpU2 = u2[index + ((STENCIL + 1) * dimZX)];
        index += dimZX;

        //////////////////////////////////////////
        // advance the slice (move the thread-front)
        back.w = back.z;
        back.z = back.y;
        back.y = back.x;
        back.x = current;
        current = front.w;
        front.w = front.z;
        front.z = front.y;
        front.y = front.x;
        front.x = tmpU2;

        __syncthreads();

        /////////////////////////////////////////
        // update the data slice in smem
        //s_data[SH(tx + (STENCIL * signX), ty)] = u2[index + STENCIL * signX];
        //s_data[SH(tx, ty + signY * STENCIL)] = u2[index + dimZ) * signY * STENCIL];

        if (threadIdx.x < STENCIL) { // halo left/right
            s_data[SH(threadIdx.x, ty)            ] = u2[index - STENCIL];
            s_data[SH(tx + STENCIL_TILE_XSIZE, ty)] = u2[index + STENCIL_TILE_XSIZE];
        }
        __syncthreads();
        if (threadIdx.y < STENCIL) { // halo above/below
            s_data[SH(tx, threadIdx.y)            ] = u2[index - STENCIL            * dimZ];
            s_data[SH(tx, ty + STENCIL_TILE_YSIZE)] = u2[index + STENCIL_TILE_YSIZE * dimZ];
        }

        /////////////////////////////////////////
        // compute the output value
        s_data[SH(tx, ty)] = current;
        __syncthreads();
        float tmp  = v[realIndex];
        float tmp1 = u1[index];

        float div  =
            devX4 * (s_data[SH_Y(-4)] + s_data[SH_Y(4)]);
        div += devC00 * current;
        div += devX3 * (s_data[SH_Y(-3)] + s_data[SH_Y(3)]);
        div += devX2 * (s_data[SH_Y(-2)] + s_data[SH_Y(2)]);
        div += devX1 * (s_data[SH_Y(-1)] + s_data[SH_Y(1)]);
        div += devY4 * (front.x + back.w);
        div += devZ4 * (s_data[SH_X(-4)] + s_data[SH_X(4)]);
        div += devY3 * (front.y + back.z);
        div += devZ3 * (s_data[SH_X(-3)] + s_data[SH_X(3)]);
        div += devY2 * (front.z + back.y);
        div += devZ2 * (s_data[SH_X(-2)] + s_data[SH_X(2)]);
        div += devY1 * (front.w + back.x);
        div += devZ1 * (s_data[SH_X(-1)] + s_data[SH_X(1)]);

        div = tmp * tmp * div;
        div = dt2 * div + current + current - tmp1;
        u3[index] = div;

        realIndex += dimRealZX;
    }
}

int main(int argc, char *argv[])
{
	float * u1 = NULL, * u2 = NULL, * u3 = NULL, * v = NULL;
	struct timeval s, t;

	if(argv[ARGS] != NULL) {
        size_t dim = atoi(argv[ARGS]);
        if (dim % 32 != 0) {
            fprintf(stderr, "Error: wrong dimension %d\n", dim);
            abort();
        }
        dimRealElems = dim;
        dimElems     = dim + 2 * STENCIL;
    }

	srand(time(NULL));

	gettimeofday(&s, NULL);

	// Alloc 3 volumes for 2-degree time integration
	if(gmacMalloc((void **)&u1, SIZE_3D(dimElems)) != gmacSuccess)
		CUFATAL();
    gmacMemset(u1, 0, SIZE_3D(dimElems));	
	if(gmacMalloc((void **)&u2, SIZE_3D(dimElems)) != gmacSuccess)
		CUFATAL();
	gmacMemset(u2, 0, SIZE_3D(dimElems));
	if(gmacMalloc((void **)&u3, SIZE_3D(dimElems)) != gmacSuccess)
		CUFATAL();
    gmacMemset(u3, 0, SIZE_3D(dimElems));
    for (int k = (dimElems/2) - 2; k <= (dimElems/2) + 2; k++) {        
        for (int j = (dimElems/2) - 2; j <= (dimElems/2) + 2; j++) {        
            for (int i = (dimElems/2) - 2; i <= (dimElems/2) + 2; i++) {        
                int iter = k * ELEMS_2D(dimElems) + j * dimElems + i;
                u3[iter] = INIT_VAL;
            }
        }
    }

    if(gmacMalloc((void **)&v, SIZE_3D(dimRealElems)) != gmacSuccess)
		CUFATAL();

    for (int k = 0; k < dimRealElems; k++) {        
        for (int j = 0; j < dimRealElems; j++) {        
            for (int i = 0; i < dimRealElems; i++) {        
                int iter = k * ELEMS_2D(dimRealElems) + j * dimRealElems + i;
                v[iter] = VELOCITY;
            }
        }
    }

	gettimeofday(&t, NULL);
	printTime(&s, &t, "Alloc: ", "\n");

	// Call the kernel
	dim3 Db(32, 8);
	dim3 Dg(dimElems / 32, dimElems / 8);
	gettimeofday(&s, NULL);
    for (uint32_t i = 0; i < ITERATIONS; i++) {
        float * tmp;

        kernelStencil<32, 8><<<Dg, Db>>>(gmacPtr(u1), gmacPtr(u2), gmacPtr(u3), gmacPtr(v), 0.08, dimElems, dimRealElems, ELEMS_2D(dimElems), ELEMS_2D(dimRealElems), dimRealElems);
        if(gmacThreadSynchronize() != gmacSuccess) CUFATAL();

        tmp = u3;
        u3 = u1;
        u1 = u2;
        u2 = tmp;
    }

	gettimeofday(&t, NULL);
	printTime(&s, &t, "Run: ", "\n");

	gmacFree(u1);
	gmacFree(u2);
	gmacFree(u3);

	gmacFree(v);
}
