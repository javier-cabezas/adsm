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

#define ITERATIONS 250

float doTest(float *a, float *b)
{
	gmactime_t s, t;
    char path_base[] = "/tmp/_gmac_file_";
    char path[256];

    for (int i = 0; i < ITERATIONS; i++) {
        memset(path, 0, 256);
        sprintf(path, "%s%d", path_base, i);

        dim3 Db(blockSize);
        dim3 Dg(vecSize / blockSize);
        if(vecSize % blockSize) Dg.x++;
        vecSet<<<Dg, Db>>>(gmacPtr(a), vecSize, float(i));
        gmacThreadSynchronize();

        FILE * f = fopen(path, "wb");
        assert(f != NULL);
        assert(fwrite(a, sizeof(float), vecSize, f) == vecSize);
        fclose(f);
    }

    for (int i = ITERATIONS - 1; i >= 0; i--) {
        memset(path, 0, 256);
        sprintf(path, "%s%d", path_base, i);

        FILE * f = fopen(path, "rb");
        assert(f != NULL);
        assert(fread(a, sizeof(float), vecSize, f) == vecSize);
        fclose(f);

        dim3 Db(blockSize);
        dim3 Dg(vecSize / blockSize);
        if(vecSize % blockSize) Dg.x++;
        vecAccum<<<Dg, Db>>>(gmacPtr(b), gmacPtr(a), vecSize);
        gmacThreadSynchronize();
    }

    float error = 0;
    for(unsigned i = 0; i < vecSize; i++) {
        error += b[i] - (ITERATIONS - 1)*(ITERATIONS / 2);
    }
    fprintf(stderr,"Error: %f\n", error);
    return error;
}

int main(int argc, char *argv[])
{
	float *a, *b;
	gmactime_t s, t;
    float error1, error2, error3;

	fprintf(stdout, "Vector: %f\n", 1.0 * vecSize / 1024 / 1024);

    //////////////////////
    // Test shared objects
    //////////////////////
    fprintf(stderr,"SHARED OBJECTS\n");
    getTime(&s);
    // Alloc & init input data
    if(gmacMalloc((void **)&a, vecSize * sizeof(float)) != gmacSuccess)
        CUFATAL();
    if(gmacMalloc((void **)&b, vecSize * sizeof(float)) != gmacSuccess)
        CUFATAL();
    getTime(&t);
    printTime(&s, &t, "Alloc: ", "\n");

    gmacMemset(a, 0, vecSize * sizeof(float));
    gmacMemset(b, 0, vecSize * sizeof(float));

    error1 = doTest(a, b);

    gmacFree(a);
    gmacFree(b);
    return error1 != 0.f;

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
