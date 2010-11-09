#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gmac.h>

#include "utils.h"
#include "debug.h"

#ifdef _MSC_VER
#define VECTORA "inputset\\vectorA"
#define VECTORB "inputset\\vectorB"
#define VECTORC "inputset\\vectorC"
#else
#define VECTORA "inputset/vectorA"
#define VECTORB "inputset/vectorB"
#define VECTORC "inputset/vectorC"
#endif

const size_t vecSize = 1024 * 1024;
const size_t blockSize = 512;

const char *msg = "Done!";

__global__ void vecSet(float *a, size_t size, float val)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= size) return;

    a[i] = float(val);
}

__global__ void vecAccum(float *b, const float *a, size_t size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= size) return;

    b[i] += a[i];
}

__global__ void vecMove(float *a, const float *b, size_t size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= size) return;

    a[i] = b[i];
}

#define ITERATIONS 250

pthread_barrier_t ioAfter;
pthread_barrier_t ioBefore;

float *a, *b, *c;

float error1, error2;

void *doTest(void *)
{
    // Alloc & init input data
    if(gmacMalloc((void **)&a, vecSize * sizeof(float)) != gmacSuccess)
        CUFATAL();
    if(gmacMalloc((void **)&b, vecSize * sizeof(float)) != gmacSuccess)
        CUFATAL();
    if(gmacMalloc((void **)&c, vecSize * sizeof(float)) != gmacSuccess)
        CUFATAL();

    gmacMemset(a, 0, vecSize * sizeof(float));
    gmacMemset(b, 0, vecSize * sizeof(float));

    pthread_barrier_wait(&ioBefore);

	gmactime_t s, t;
    dim3 Db(blockSize);
    dim3 Dg(vecSize / blockSize);
    if(vecSize % blockSize) Dg.x++;

    for (int i = 0; i < ITERATIONS; i++) {
        vecSet<<<Dg, Db>>>(gmacPtr(a), vecSize, float(i));
        pthread_barrier_wait(&ioAfter);
        vecMove<<<Dg, Db>>>(gmacPtr(c), gmacPtr(a), vecSize);
        gmacThreadSynchronize();
        pthread_barrier_wait(&ioBefore);
    }

    pthread_barrier_wait(&ioBefore);

    for (int i = ITERATIONS - 1; i >= 0; i--) {
        pthread_barrier_wait(&ioBefore);
        vecAccum<<<Dg, Db>>>(gmacPtr(b), gmacPtr(a), vecSize);
        gmacThreadSynchronize();
        pthread_barrier_wait(&ioAfter);
    }


    error1 = 0.f;
    for(unsigned i = 0; i < vecSize; i++) {
        error1 += b[i] - (ITERATIONS - 1)*(ITERATIONS / 2);
    }
    fprintf(stderr,"Error: %f\n", error1);

    gmacFree(a);
    gmacFree(b);

    return &error1;
}

void *doTestIO(void *)
{
	gmactime_t s, t;
    char path_base[] = "/tmp/_gmac_file_";
    char path[256];

    pthread_barrier_wait(&ioBefore);

    for (int i = 0; i < ITERATIONS; i++) {
        pthread_barrier_wait(&ioAfter);
        pthread_barrier_wait(&ioBefore);
        memset(path, 0, 256);
        sprintf(path, "%s%d", path_base, i);

        FILE * f = fopen(path, "wb");
        assert(f != NULL);
        assert(fwrite(c, sizeof(float), vecSize, f) == vecSize);
        fclose(f);
    }

    pthread_barrier_wait(&ioBefore);

    for (int i = ITERATIONS - 1; i >= 0; i--) {
        memset(path, 0, 256);
        sprintf(path, "%s%d", path_base, i);

        FILE * f = fopen(path, "rb");
        assert(f != NULL);
        assert(fread(a, sizeof(float), vecSize, f) == vecSize);
        fclose(f);

        pthread_barrier_wait(&ioBefore);
        pthread_barrier_wait(&ioAfter);
    }

    return &error2;
}

int main(int argc, char *argv[])
{
	gmactime_t s, t;
    pthread_t tid, tidIO;
    float *error;

	fprintf(stdout, "Vector: %f\n", 1.0 * vecSize / 1024 / 1024);

    pthread_barrier_init(&ioAfter,  NULL, 2);
    pthread_barrier_init(&ioBefore, NULL, 2);

    pthread_create(&tid, NULL, doTest, NULL);
    pthread_create(&tidIO, NULL, doTestIO, NULL);

    pthread_join(tid, (void **) &error);
    pthread_join(tidIO, NULL);

    pthread_barrier_destroy(&ioAfter);
    pthread_barrier_destroy(&ioBefore);

    return *error != 0.f;

#if 0
    //////////////////////////
    // Test replicated objects
    //////////////////////////
    fprintf(stderr,"REPLICATED OBJECTS\n");
    getTime(&s);
    // Alloc & init input data
    if(gmacGlobalMalloc((void **)&a, vecSize * sizeof(float), GMAC_GLOBAL_MALLOC_REPLICATED) != gmacSuccess)
        CUFATAL();
    if(gmacGlobalMalloc((void **)&b, vecSize * sizeof(float), GMAC_GLOBAL_MALLOC_REPLICATED) != gmacSuccess)
        CUFATAL();
    getTime(&t);
    printTime(&s, &t, "Alloc: ", "\n");

    error2 = doTest(a, b, c, orig);

    fC = fopen("vectorC_replicated", "w");
    fwrite(c, sizeof(float), vecSize, fC);
    fclose(fC);

    gmacFree(a);
    gmacFree(b);

    ///////////////////////////
    // Test centralized objects
    ///////////////////////////
    fprintf(stderr,"CENTRALIZED OBJECTS\n");
    getTime(&s);
    // Alloc & init input data
    if(gmacGlobalMalloc((void **)&a, vecSize * sizeof(float), GMAC_GLOBAL_MALLOC_CENTRALIZED) != gmacSuccess)
        CUFATAL();
    if(gmacGlobalMalloc((void **)&b, vecSize * sizeof(float), GMAC_GLOBAL_MALLOC_CENTRALIZED) != gmacSuccess)
        CUFATAL();
    getTime(&t);
    printTime(&s, &t, "Alloc: ", "\n");

    error3 = doTest(a, b, c, orig);

    fC = fopen("vectorC_centralized", "w");
    fwrite(c, sizeof(float), vecSize, fC);
    fclose(fC);

    gmacFree(a);
    gmacFree(b);

    gmacFree(c);
    free(orig);
    return error1 != 0.f && error2 != 0.f && error3 != 0.f;
#endif
}
