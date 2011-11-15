#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include <gmac/cl.h>

#include "clBitonicSortKernel.cl"

#define GROUP_SIZE 1

void
swapIfFirstIsGreater(cl_uint *a, cl_uint *b)
{
    if(*a > *b) {
        cl_uint temp = *a;
        *a = *b;
        *b = temp;
    }
}

/*
 * sorts the input array (in place) using the bitonic sort algorithm
 * sorts in increasing order if sortIncreasing is CL_TRUE
 * else sorts in decreasing order
 * length specifies the length of the array
 */
void
bitonicSortCPUReference(
    cl_uint *input,
    const cl_uint length,
    const cl_bool sortIncreasing)
{
    const cl_uint halfLength = length/2;

    cl_uint i;
    for(i = 2; i <= length; i *= 2) {
        cl_uint j;
        for(j = i; j > 1; j /= 2) {
            cl_bool increasing = sortIncreasing;
            const cl_uint half_j = j/2;

            cl_uint k;
            for(k = 0; k < length; k += j) {
                const cl_uint k_plus_half_j = k + half_j;
                cl_uint l;

                if(i < length) {
                    if(((k == i) || ((k % i) == 0)) && (k != halfLength))
                        increasing = !increasing;
                }

                for(l = k; l < k_plus_half_j; ++l) {
                    if(increasing)
                        swapIfFirstIsGreater(&input[l], &input[l + half_j]);
                    else
                        swapIfFirstIsGreater(&input[l + half_j], &input[l]);
                }
            }
        }
    }
}

int main(int argc, char *argv[])
{
    cl_platform_id platform;
    cl_device_id device;
    cl_int error_code;
    cl_context context;
    cl_command_queue command_queue;
    cl_program program;
    cl_kernel kernel;

    cl_uint seed = 123;
    cl_uint sortDescending = 0;
    cl_uint *input = NULL;
    cl_uint *verificationInput;
    cl_uint length = 1024;

    error_code = clGetPlatformIDs(1, &platform, NULL);
    assert(error_code == CL_SUCCESS);
    error_code = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    assert(error_code == CL_SUCCESS);
    context = clCreateContext(0, 1, &device, NULL, NULL, &error_code);
    assert(error_code == CL_SUCCESS);
    command_queue = clCreateCommandQueue(context, device, 0, &error_code);
    assert(error_code == CL_SUCCESS);
    program = clCreateProgramWithSource(context, 1, &code, NULL, &error_code);
    assert(error_code == CL_SUCCESS);
    error_code = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    assert(error_code == CL_SUCCESS);
    kernel = clCreateKernel(program, "bitonicSort", &error_code);
    assert(error_code == CL_SUCCESS);

    // Alloc
    error_code = clMalloc(command_queue, (void **)&input, length * sizeof(cl_uint));
	assert(error_code == CL_SUCCESS);
    verificationInput = (cl_uint *) malloc(length*sizeof(cl_uint));
    if(verificationInput == NULL)
        return 0;
   
    /* random initialisation of input */
    srand(seed);
    const cl_uint rangeMin = 0;
    const cl_uint rangeMax = 255;
    double range = double(rangeMax - rangeMin) + 1.0;
    for(cl_uint i = 0; i < length; i++) {
        input[i] = rangeMin + (cl_uint)(range * rand() / (RAND_MAX + 1.0));
    }
    memcpy(verificationInput, input, length*sizeof(cl_uint));

    // Print the input data
    printf("Unsorted Input: ");
    for(cl_uint i = 0; i < length; i++)
        printf("%d ", input[i]);

    cl_uint numStages = 0;
    size_t globalThreads[1] = {length / 2};
    size_t localThreads[1] = {GROUP_SIZE};

    for(cl_uint temp = length; temp > 1; temp >>= 1)
        ++numStages;
    cl_mem input_device = clGetBuffer(context, input);
    error_code = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_device);
	assert(error_code == CL_SUCCESS);
    error_code = clSetKernelArg(kernel, 3, sizeof(length), &length);
	assert(error_code == CL_SUCCESS);
    error_code = clSetKernelArg(kernel, 4, sizeof(sortDescending), &sortDescending);
	assert(error_code == CL_SUCCESS);
    for(cl_uint stage = 0; stage < numStages; ++stage) {
        /* stage of the algorithm */
        error_code = clSetKernelArg(kernel, 1, sizeof(stage), &stage);
		assert(error_code == CL_SUCCESS);
        /* Every stage has stage+1 passes. */
        for(cl_uint passOfStage = 0; passOfStage < stage + 1; ++passOfStage) {
            /* pass of the current stage */
            error_code = clSetKernelArg(kernel, 2, sizeof(passOfStage), &passOfStage);
			assert(error_code == CL_SUCCESS);
            error_code = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, globalThreads, localThreads, 0, NULL, NULL);
			assert(error_code == CL_SUCCESS);
            error_code = clFinish(command_queue);
			assert(error_code == CL_SUCCESS);
        }
    }

    printf("Output: ");
    for(cl_uint i = 0; i < length; i++) {
        printf("%d ", input[i]);
    }

    bitonicSortCPUReference(verificationInput, length, sortDescending);
    if(memcmp(input, verificationInput, length*sizeof(cl_uint)) == 0) {
        printf("\nPassed!\n");
    } else {
        printf("\nFailed\n");
    }

    free(verificationInput);
    verificationInput = NULL;
    assert(clFree(command_queue, input) == CL_SUCCESS);

    return 0;
}
