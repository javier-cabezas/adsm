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
// declaration, forward
void runTest(int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv)
{
    runTest(argc, argv);

    return 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest(int argc, char** argv)
{
    // allocate host memory for matrices A and B
    unsigned int elemsA = WA * HA;
    unsigned int sizeA = sizeof(float) * elemsA;
    unsigned int elemsB = WB * HB;
    unsigned int sizeB = sizeof(float) * elemsB;

    // create and start timer
    struct timeval s, t;
    gettimeofday(&s, NULL);

    // allocate device memory
    float* A;
    if (gmacMalloc((void**) &A, sizeA) != gmacSuccess) {
        fprintf(stderr, "Error allocating A");
        abort();
    }
    float* B;
    if (gmacMalloc((void**) &B, sizeB) != gmacSuccess) {
        fprintf(stderr, "Error allocating B");
        abort();
    }

    // initialize host memory
    randInit(A, elemsA);
    randInit(B, elemsB);

    // allocate device memory for result
    unsigned int elemsC = WC * HC;
    unsigned int sizeC = sizeof(float) * elemsC;
    float* C;
    if (gmacMalloc((void**) &C, sizeC) != gmacSuccess) {
        fprintf(stderr, "Error allocating A");
        abort();
    }
	gettimeofday(&t, NULL);
	printTime(&s, &t, "Alloc: ", "\n");

    // Call the kernel
	gettimeofday(&s, NULL);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(WC / threads.x, HC / threads.y);
    matrixMul<<< grid, threads >>>(C, A, B, WA, WB);
	if(gmacThreadSynchronize() != gmacSuccess) CUFATAL();
	gettimeofday(&t, NULL);
	printTime(&s, &t, "Run: ", "\n");


    // compute reference solution
	gettimeofday(&s, NULL);
    float* reference = (float *) malloc(sizeC);
    computeGold(reference, A, B, HA, WA, WB);

    // check result
    float err;
    //err = checkError(reference, C, elemsC);
	gettimeofday(&t, NULL);
	printTime(&s, &t, "Check: ", "\n");

    fprintf(stderr, "Error: %f\n", err);

    // clean up memory
    gmacFree(A);
    gmacFree(B);
    gmacFree(C);
    free(reference);

    return;
}
