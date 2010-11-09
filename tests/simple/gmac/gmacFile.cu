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

static void
writeFile(float *v, unsigned nmemb, int it);
static void
readFile(float *v, unsigned nmemb, int it);

float *a, *b, *c;

float error1, error2;

void *doTest(void *)
{
	gmactime_t s, t;
    dim3 Db(blockSize);
    dim3 Dg(vecSize / blockSize);

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
        pthread_barrier_wait(&ioAfter);
        //readFile(a, vecSize, i);
        vecAccum<<<Dg, Db>>>(gmacPtr(b), gmacPtr(a), vecSize);
        gmacThreadSynchronize();
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

static void
setPath(char *name, size_t len, int it)
{
    static const char path_base[] = "/tmp/_gmac_file_";
    memset(name, '\0', len);
    sprintf(name, "%s%d", path_base, it);
}

static void
writeFile(float *v, unsigned nmemb, int it)
{
    char path[256];
    setPath(path, 256, it);

    FILE * f = fopen(path, "wb");
    assert(f != NULL);
    assert(fwrite(v, sizeof(float), nmemb, f) == nmemb);
    fclose(f);
}

static void
readFile(float *v, unsigned nmemb, int it)
{
    char path[256];
    setPath(path, 256, it);

    FILE * f = fopen(path, "rb");
    assert(f != NULL);
    assert(fread(v, sizeof(float), nmemb, f) == nmemb);
    fclose(f);
}

void *doTestIO(void *)
{
	gmactime_t s, t;

    pthread_barrier_wait(&ioBefore);

    for (int i = 0; i < ITERATIONS; i++) {
        pthread_barrier_wait(&ioAfter);
        pthread_barrier_wait(&ioBefore);
        writeFile(c, vecSize, i);
    }

    pthread_barrier_wait(&ioBefore);

    for (int i = ITERATIONS - 1; i >= 0; i--) {
        pthread_barrier_wait(&ioBefore);
        readFile(a, vecSize, i);
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
}
