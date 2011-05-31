#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <gmac/opencl.h>

#include "utils.h"

enum MemcpyType {
    GMAC_TO_GMAC = 1,
    HOST_TO_GMAC = 2,
    GMAC_TO_HOST = 3,
};

int type;
int typeDefault = GMAC_TO_GMAC;
const char *typeStr = "GMAC_MEMCPY_TYPE";

bool memcpyFn;
bool memcpyFnDefault = false;
const char *memcpyFnStr = "GMAC_MEMCPY_GMAC";

const size_t minCount = 1024;
const size_t maxCount = 16 * 1024 * 1024;

const char *kernel = "\
__kernel void null()\
{\
	return;\
}\
";

void init(uint8_t *ptr, int s, uint8_t v)
{
	for(int i = 0; i < s; i++) {
		ptr[i] = v;
	}
}

int memcpyTest(MemcpyType type, bool callKernel, void *(*memcpy_fn)(void *, const void *, size_t n))
{
    int error = 0;

    ocl_kernel kernel;
    size_t globalSize = 1;
    size_t localSize = 1;

    assert(oclGetKernel("null", &kernel) == oclSuccess);

    uint8_t *baseSrc = NULL;
    uint8_t *oclSrc = NULL;
    uint8_t *oclDst = NULL;

    baseSrc = (uint8_t *)malloc(maxCount);
    init(baseSrc, int(maxCount), 0xca);
    for (size_t count = minCount; count <= maxCount; count *= 2) {
        fprintf(stderr, "ALLOC: "FMT_SIZE"\n", count);

        if (type == GMAC_TO_GMAC) {
            assert(oclMalloc((void **)&oclSrc, count) == oclSuccess);
            assert(oclMalloc((void **)&oclDst, count) == oclSuccess);
        } else if (type == HOST_TO_GMAC) {
            oclSrc = (uint8_t *)malloc(count);
            assert(oclMalloc((void **)&oclDst, count) == oclSuccess);
        } else if (type == GMAC_TO_HOST) {
            assert(oclMalloc((void **)&oclSrc, count) == oclSuccess);
            oclDst = (uint8_t *)malloc(count);
        }

        for (size_t stride = 0, i = 1; stride < count/3; stride = i, i =  i * 2 - (i == 1? 0: 1)) {
            for (size_t copyCount = 1; copyCount < count/3; copyCount *= 2) {
                init(oclSrc + stride, int(copyCount), 0xca);
                if (stride == 0) {
                    init(oclDst + stride, int(copyCount) + 1, 0);
                } else {
                    init(oclDst + stride - 1, int(copyCount) + 2, 0);
                }
                assert(stride + copyCount <= count);

                if (callKernel) {
                    assert(oclCallNDRange(kernel, 1, NULL, &globalSize, &localSize) == oclSuccess);
                }
                memcpy_fn(oclDst + stride, oclSrc + stride, copyCount);

                int ret = memcmp(oclDst + stride, baseSrc + stride, copyCount);
                if (stride == 0) {
                ret = ret && (oclDst[stride - 1] == 0 && oclDst[stride + copyCount] == 0);
                } else {
                    ret = ret && (oclDst[stride - 1] == 0 && oclDst[stride + copyCount] == 0);
                }

                if (ret != 0) {
#if 0
                    fprintf(stderr, "Error: oclToGmacTest size: %zd, stride: %zd, copy: %zd\n",
                            count    ,
                            stride   ,
                            copyCount);
#endif
                    error = 1;
                    goto exit_test;
                }
#if 0
                for (unsigned k = 0; k < count; k++) {
                    int ret = baseDst[k] != oclDst[k];
                    if (ret != 0) {
                        fprintf(stderr, "Error: oclToGmacTest size: %zd, stride: %zd, copy: %zd. Pos %u\n", count    ,
                                stride   ,
                                copyCount, k);
                        error = 1;
                    }
                }
#endif
            }
        }

        if (type == GMAC_TO_GMAC) {
            assert(oclFree(oclSrc) == oclSuccess);
            assert(oclFree(oclDst) == oclSuccess);
        } else if (type == HOST_TO_GMAC) {
            free(oclSrc);
            assert(oclFree(oclDst) == oclSuccess);
        } else if (type == GMAC_TO_HOST) {
            assert(oclFree(oclSrc) == oclSuccess);
            free(oclDst);
        }
    }
    free(baseSrc);

    oclReleaseKernel(kernel);

    return error;

exit_test:
    if (type == GMAC_TO_GMAC) {
        assert(oclFree(oclSrc) == oclSuccess);
        assert(oclFree(oclDst) == oclSuccess);
    } else if (type == HOST_TO_GMAC) {
        free(oclSrc);
        assert(oclFree(oclDst) == oclSuccess);
    } else if (type == GMAC_TO_HOST) {
        assert(oclFree(oclSrc) == oclSuccess);
        free(oclDst);
    }

    free(baseSrc);

    return error;
}

static void *oclMemcpyWrapper(void *dst, const void *src, size_t size)
{
	return oclMemcpy(dst, src, size);
}

int main(int argc, char *argv[])
{
	setParam<int>(&type, typeStr, typeDefault);
	setParam<bool>(&memcpyFn, memcpyFnStr, memcpyFnDefault);

    assert(oclCompileSource(kernel) == oclSuccess);

    int ret;
    
    if (memcpyFn == true) {
        ret = memcpyTest(MemcpyType(type), false, oclMemcpyWrapper);
        if (ret == 0) ret = memcpyTest(MemcpyType(type), true, oclMemcpyWrapper);
    } else {
        ret = memcpyTest(MemcpyType(type), false, memcpy);
        if (ret == 0) ret = memcpyTest(MemcpyType(type), true, memcpy);
    }

    return ret;
}
