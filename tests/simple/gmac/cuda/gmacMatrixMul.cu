#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>

#include <gmac/cuda.h>

#include "debug.h"
#include "utils.h"

#include "gmacMatrixMulKernel.cu"

const char * nIterStr = "GMAC_NITER";
const char * WAStr = "GMAC_WA";
const char * HAStr = "GMAC_HA";
const char * WBStr = "GMAC_WB";
const char * HBStr = "GMAC_HB";
const char * centralObjectsStr = "GMAC_CENTRALIZED";

const int nIterDefault = 4;
const int WADefault = (40 * BLOCK_SIZE); // Matrix A width
const int HADefault = (40 * BLOCK_SIZE); // Matrix A height
const int WBDefault = (40 * BLOCK_SIZE); // Matrix B width
const int HBDefault = (40 * BLOCK_SIZE); // Matrix B height
const int centralObjectsDefault = 0;

static int nIter = 0;
static int WA = 0; // Matrix A width
static int HA = 0; // Matrix A height
static int WB = 0; // Matrix B width
static int HB = 0; // Matrix B height
static int centralObjects = 0;

#define WC WB  // Matrix C width 
#define HC HA  // Matrix C height

static float * A, * B;
struct param {
	int i;
	float * ptr;
};

unsigned elemsC;
unsigned sizeC;

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


int
main(int argc, char** argv)
{
	setParam<int>(&nIter, nIterStr, nIterDefault);
	setParam<int>(&WA, WAStr, WADefault);
	setParam<int>(&HA, HAStr, HADefault);
	setParam<int>(&WB, WBStr, WBDefault);
	setParam<int>(&HB, HBStr, HBDefault);
    setParam<int>(&centralObjects, centralObjectsStr, centralObjectsDefault);

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

    GmacGlobalMallocType allocFlags;
    if(centralObjects == 1) allocFlags = GMAC_GLOBAL_MALLOC_CENTRALIZED;
    else allocFlags = GMAC_GLOBAL_MALLOC_REPLICATED;
    
    gmactime_t s, t;

    // allocate memory for matrices A and B
	getTime(&s);
    if (gmacGlobalMalloc((void**) &A, sizeA, allocFlags) != gmacSuccess) {
        fprintf(stderr, "Error allocating A");
        abort();
    }
    if (gmacGlobalMalloc((void**) &B, sizeB, allocFlags) != gmacSuccess) {
        fprintf(stderr, "Error allocating B");
        abort();
    }
    getTime(&t);
    printTime(&s, &t, "Alloc: ", "\n");

    getTime(&s);
    float err = doTest(A, B, elemsA, elemsB, elemsC);
	getTime(&t);
	printTime(&s, &t, "Total: ", "\n");

    assert(gmacFree(A) == gmacSuccess);
	assert(gmacFree(B) == gmacSuccess);

    return fabsf(err);
}
