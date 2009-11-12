/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/* Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication and is exactly the same as
 * Chapter 7 of the programming guide.
 * It has been written for clarity of exposition to illustrate various CUDA
 * programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication.
 *
 * CUBLAS provides high-performance matrix multiplication.
 */

// includes, system
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>

extern "C" {
#include <pthread.h>
}

// includes, project
#include <gmac.h>

// includes, utils and debug
#include "debug.h"
#include "utils.h"

// includes, kernels
#include "gmacMatrixMulKernel.cu"

const char * nIterStr = "GMAC_NITER";
const char * WAStr = "GMAC_WA";
const char * HAStr = "GMAC_HA";
const char * WBStr = "GMAC_WB";
const char * HBStr = "GMAC_HB";

const size_t nIterDefault = 4;
const size_t WADefault = (100 * BLOCK_SIZE); // Matrix A width
const size_t HADefault = (100 * BLOCK_SIZE); // Matrix A height
const size_t WBDefault = (100 * BLOCK_SIZE); // Matrix B width
const size_t HBDefault = (100 * BLOCK_SIZE); // Matrix B height

static size_t nIter = 0;
static size_t WA = 0; // Matrix A width
static size_t HA = 0; // Matrix A height
static size_t WB = 0; // Matrix B width
static size_t HB = 0; // Matrix B height

#define WC WB  // Matrix C width 
#define HC HA  // Matrix C height

static float * A, * B;
struct param {
	int i;
	float * ptr;
};

unsigned elemsC;
unsigned sizeC;

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! C = A * B
//! @param C          reference data, computed but preallocated
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param hA         height of matrix A
//! @param wB         width of matrix B
////////////////////////////////////////////////////////////////////////////////
void
computeGold(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j) {
            double sum = 0;
            for (unsigned int k = 0; k < wA; ++k) {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }
            C[i * wB + j] = (float)sum;
        }
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void *
matrixMulThread(void * ptr)
{
	struct param *p = (struct param *) ptr;

    // timers
    struct timeval s, t;

    if (gmacMalloc((void**) &p->ptr, sizeC) != gmacSuccess) {
        fprintf(stderr, "Error allocating C");
        abort();
    }

    // Call the kernel
	gettimeofday(&s, NULL);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(WC / threads.x, (HC / nIter) / threads.y);
    matrixMul<<< grid, threads >>>(gmacPtr(p->ptr), gmacPtr(A), gmacPtr(B), WA, WB, p->i * elemsC);
	if(gmacThreadSynchronize() != gmacSuccess) CUFATAL();
	gettimeofday(&t, NULL);
	printTime(&s, &t, "Run: ", "\n");

    return NULL;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv)
{
	setParam<size_t>(&nIter, nIterStr, nIterDefault);
	setParam<size_t>(&WA, WAStr, WADefault);
	setParam<size_t>(&HA, HAStr, HADefault);
	setParam<size_t>(&WB, WBStr, WBDefault);
	setParam<size_t>(&HB, HBStr, HBDefault);

    if (nIter == 0) {
        fprintf(stderr, "Error: nIter should be greater than 0\n");
        abort();
    }


    struct timeval s, t;

    pthread_t * threads = new pthread_t[nIter];
	param * params = new param[nIter];

    unsigned elemsA = WA * HA;
    unsigned elemsB = WB * HB;
             elemsC = WC * HC / nIter;
    unsigned sizeA = sizeof(float) * elemsA;
    unsigned sizeB = sizeof(float) * elemsB;
             sizeC = sizeof(float) * elemsC;

    printf("Elems: %d\n", elemsA);
    printf("Elems: %d\n", elemsB);
    printf("Elems: %d\n", elemsC);


    // allocate memory for matrices A and B
	gettimeofday(&s, NULL);
    if (gmacGlobalMalloc((void**) &A, sizeA) != gmacSuccess) {
        fprintf(stderr, "Error allocating A");
        abort();
    }
    if (gmacGlobalMalloc((void**) &B, sizeB) != gmacSuccess) {
        fprintf(stderr, "Error allocating B");
        abort();
    }

    // initialize matricesmatrices
    randInitMax(A, 100.f, elemsA);
    randInitMax(B, 100.f, elemsB);

	// Alloc output data
	gettimeofday(&t, NULL);
	printTime(&s, &t, "Alloc: ", "\n");

    for (unsigned n = 0; n < nIter; n++) {
		params[n].i = n;
		pthread_create(&threads[n], NULL, matrixMulThread, &(params[n]));
	}

	gmacFree(A);
	gmacFree(B);

	for (unsigned n = 0; n < nIter; n++) {
		pthread_join(threads[n], NULL);
	}

    // compute reference solution
	gettimeofday(&s, NULL);
    // check result
    float err;
    printf("Computing host matrix mul. Please wait...\n");
    float* reference = (float *) malloc(sizeC * nIter);
    computeGold(reference, A, B, HA, WA, WB);

    for (unsigned n = 0; n < nIter; n++) {
        err += checkError(reference + n * elemsC, params[n].ptr, elemsC);
	}
	gettimeofday(&t, NULL);
	printTime(&s, &t, "Check: ", "\n");

    fprintf(stderr, "Error: %f\n", err);
    // clean up memory
    free(reference);

    delete [] params;
    delete [] threads;

    return 0;
}
