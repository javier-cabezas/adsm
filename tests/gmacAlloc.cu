#include <stdio.h>
#include <gmac.h>

__global__ void kernelFill(int *A, off_t off, size_t size)
{
    int localIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = localIdx + off;

    if (idx >= size) return;
    A[localIdx] = idx;
}

int main(int argc, char *argv[])
{
    const size_t totalSize = 8 * 1024 * 1024;
    for (size_t currentSize = totalSize; currentSize > 0; currentSize /= 2) {
        fprintf(stderr, "Testing object size %zd\n", currentSize);
        assert(totalSize % currentSize == 0);
        size_t nObjects = totalSize / currentSize;

        int **objects = (int **) malloc(nObjects * sizeof(int *));
        assert(objects != NULL);

        fprintf(stderr, "- Allocating: %zd objects\n", nObjects);
        for(int i = 0; i < nObjects; i++) {
            assert(gmacMalloc((void **)&objects[i], currentSize * sizeof(int)) == gmacSuccess);
        }

        fprintf(stderr, "- Running kernel\n");
        size_t off = 0;

        dim3 Db(currentSize > 256? 256: currentSize);
        dim3 Dg(currentSize / Db.x);
        if (currentSize > 256 && currentSize % 256 != 0) Dg.x++;

        for(int i = 0; i < nObjects; i++) {
            kernelFill<<<Dg, Db>>>(gmacPtr(objects[i]), off, totalSize);
            off += currentSize;
        }
        gmacThreadSynchronize();

        fprintf(stderr, "- Checking\n");
        off = 0;
        for(int i = 0; i < nObjects; i++) {
            for(int j = 0; j < currentSize; j++) {
                int idx = off + j;
                assert(objects[i][j] == idx);
            }
            off += currentSize;
        }

        fprintf(stderr, "- Freeing: %zd objects\n", nObjects);
        for(int i = 0; i < nObjects; i++) {
            gmacFree(objects[i]);
        }

        free(objects);
    }

    return 0;
}
