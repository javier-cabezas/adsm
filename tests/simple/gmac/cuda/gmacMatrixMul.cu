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

const int nIterDefault = 4;
const int WADefault = (40 * BLOCK_SIZE); // Matrix A width
const int HADefault = (40 * BLOCK_SIZE); // Matrix A height
const int WBDefault = (40 * BLOCK_SIZE); // Matrix B width
const int HBDefault = (40 * BLOCK_SIZE); // Matrix B height

static int nIter = 0;
static int WA = 0; // Matrix A width
static int HA = 0; // Matrix A height
static int WB = 0; // Matrix B width
static int HB = 0; // Matrix B height

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
    gmactime_t s, t;

    if (gmacMalloc((void**) &p->ptr, sizeC) != gmacSuccess) {
        fprintf(stderr, "Error allocating C");
        abort();
    }

    // Call the kernel
	getTime(&s);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(WC / threads.x, (HC / nIter) / threads.y);
    matrixMul<<< grid, threads >>>(gmacPtr(p->ptr), gmacPtr(A), gmacPtr(B), WA, WB, p->i * elemsC);
	if(gmacThreadSynchronize() != gmacSuccess) CUFATAL();
	getTime(&t);
	printTime(&s, &t, "Run: ", "\n");

    return NULL;
}

float doTest(float * A, float * B, unsigned elemsA, unsigned elemsB, unsigned elemsC)
{
    thread_t * threads = new thread_t[nIter];
	param * params = new param[nIter];

    gmactime_t s, t;

    // allocate memory for matrices A and B
	getTime(&s);

    // initialize matricesmatrices
    valueInit(A, 100.f, elemsA);
    valueInit(B, 100.f, elemsB);

	// Alloc output data
	getTime(&t);
	printTime(&s, &t, "Init: ", "\n");

    for (int n = 0; n < nIter; n++) {
		params[n].i = n;
		threads[n] = thread_create(matrixMulThread, &(params[n]));
	}

	for (int n = 0; n < nIter; n++) {
		thread_wait(threads[n]);
	}

    // compute reference solution
	getTime(&s);
    // check result
    float err = 0;
    printf("Computing host matrix mul. Please wait...\n");
    float* reference = (float *) malloc(sizeC * nIter);
    computeGold(reference, A, B, HA, WA, WB);

    for (int n = 0; n < nIter; n++) {
        err += checkError(reference + n * elemsC, params[n].ptr, elemsC);
    }

    getTime(&t);
    printTime(&s, &t, "Check: ", "\n");

    fprintf(stderr, "Error: %f\n", err);

    // clean up memory
    free(reference);

    for (int n = 0; n < nIter; n++) {
        assert(gmacFree(params[n].ptr) == gmacSuccess);
    }

    delete [] params;
    delete [] threads;

    return err;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv)
{
	setParam<int>(&nIter, nIterStr, nIterDefault);
	setParam<int>(&WA, WAStr, WADefault);
	setParam<int>(&HA, HAStr, HADefault);
	setParam<int>(&WB, WBStr, WBDefault);
	setParam<int>(&HB, HBStr, HBDefault);

    if (nIter == 0) {
        fprintf(stderr, "Error: nIter should be greater than 0\n");
        abort();
    }

    if ((HA/BLOCK_SIZE) % nIter != 0) {
        fprintf(stderr, "Error: wrong HA size. HA/%d nIter must be 0\n", BLOCK_SIZE);
        abort();
    }

    if (HB != WA) {
        fprintf(stderr, "Error: WA and HB must be equal\n");
        abort();
    }

    unsigned elemsA = WA * HA;
    unsigned elemsB = WB * HB;
             elemsC = WC * HC / nIter;
    unsigned sizeA = sizeof(float) * elemsA;
    unsigned sizeB = sizeof(float) * elemsB;
             sizeC = sizeof(float) * elemsC;

    
    fprintf(stderr, "Size A: %d\n", sizeA);
    fprintf(stderr, "Size B: %d\n", sizeB);
    fprintf(stderr, "Size C: %d\n", sizeC);

    gmactime_t s, t;

    /////////////////////
    // CENTRALIZED OBJECT
    /////////////////////
    fprintf(stderr, "CENTRALIZED OBJECTS\n");
    // allocate memory for matrices A and B
	getTime(&s);
    if (gmacGlobalMalloc((void**) &A, sizeA, GMAC_GLOBAL_MALLOC_CENTRALIZED) != gmacSuccess) {
        fprintf(stderr, "Error allocating A");
        abort();
    }
    if (gmacGlobalMalloc((void**) &B, sizeB, GMAC_GLOBAL_MALLOC_CENTRALIZED) != gmacSuccess) {
        fprintf(stderr, "Error allocating B");
        abort();
    }

    float err = doTest(A, B, elemsA, elemsB, elemsC);
	getTime(&t);
	printTime(&s, &t, "Total: ", "\n");

    assert(gmacFree(A) == gmacSuccess);
	assert(gmacFree(B) == gmacSuccess);

    ////////////////////
    // REPLICATED OBJECT
    ////////////////////
    fprintf(stderr, "REPLICATED OBJECTS\n");
    // allocate memory for matrices A and B
	getTime(&s);
    if (gmacGlobalMalloc((void**) &A, sizeA, GMAC_GLOBAL_MALLOC_REPLICATED) != gmacSuccess) {
        fprintf(stderr, "Error allocating A");
        abort();
    }
    if (gmacGlobalMalloc((void**) &B, sizeB, GMAC_GLOBAL_MALLOC_REPLICATED) != gmacSuccess) {
        fprintf(stderr, "Error allocating B");
        abort();
    }

    float err2 = doTest(A, B, elemsA, elemsB, elemsC);
	getTime(&t);
	printTime(&s, &t, "Total: ", "\n");

    assert(gmacFree(A) == gmacSuccess);
	assert(gmacFree(B) == gmacSuccess);

    // return fabsf(err) != 0.0f && fabsf(err2) != 0.0f;
    return fabsf(err2) != 0.0f;
}
