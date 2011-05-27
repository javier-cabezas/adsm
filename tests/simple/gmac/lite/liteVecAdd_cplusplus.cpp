#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cassert>

#include "gmac/lite"

#include "utils.h"
#include "debug.h"

const char *vecSizeStr = "GMAC_VECSIZE";
const unsigned vecSizeDefault = 32 * 1024 * 1024;
unsigned vecSize = 0;

const size_t blockSize = 32;

const char *msg = "Done!";

const char *kernel_source = "\
__kernel void vecAdd(__global float *c, __global const float *a, __global const float *b, unsigned size)\
{\
    unsigned i = get_global_id(0);\
    if(i >= size) return;\
\
    c[i] = a[i] + b[i];\
}\
";


int main(int argc, char *argv[])
{
    cl_platform_id platform;
    cl_device_id device;
    cl_int error_code;
	float *a, *b, *c;
	gmactime_t s, t;

	setParam<unsigned>(&vecSize, vecSizeStr, vecSizeDefault);
	fprintf(stdout, "Vector: %f\n", 1.0 * vecSize / 1024 / 1024);

    VECTOR_CLASS<cl::Platform> platforms;
    VECTOR_CLASS<cl::Device> devices;

    error_code = cl::Platform::get(&platforms);
    assert(error_code == CL_SUCCESS);
    error_code = platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
    assert(error_code == CL_SUCCESS);
    cl::Context context(devices, NULL, NULL, NULL, &error_code);
    assert(error_code == CL_SUCCESS);
    cl::CommandQueue command_queue(context, devices[0], NULL, &error_code);
    assert(error_code == CL_SUCCESS);
    cl::Program::Sources sources;
    sources.push_back(std::pair<const char *, ::size_t>(kernel_source, strlen(kernel_source)));
    cl::Program program(context, sources, &error_code);
    assert(error_code == CL_SUCCESS);
    error_code = program.build(devices);
    assert(error_code == CL_SUCCESS);
    cl::Kernel kernel(program, "vecAdd", &error_code);
    assert(error_code == CL_SUCCESS);

    getTime(&s);
    // Alloc & init input data
    assert(cl::malloc(context, (void **)&a, vecSize * sizeof(float)) == CL_SUCCESS);
    assert(cl::malloc(context, (void **)&b, vecSize * sizeof(float)) == CL_SUCCESS);
    // Alloc output data
    assert(cl::malloc(context, (void **)&c, vecSize * sizeof(float)) == CL_SUCCESS);
    getTime(&t);
    printTime(&s, &t, "Alloc: ", "\n");


    float sum = 0.f;

    getTime(&s);
    valueInit(a, 1.f, vecSize);
    valueInit(b, 1.f, vecSize);
    getTime(&t);
    printTime(&s, &t, "Init: ", "\n");

    for(unsigned i = 0; i < vecSize; i++) {
        sum += a[i] + b[i];
    }
    
    // Call the kernel
    getTime(&s);
    size_t local_size = blockSize;
    size_t global_size = vecSize / blockSize;
    if(vecSize % blockSize) global_size++;
    global_size *= local_size;

    assert(kernel.setArg(0, cl::getBuffer(context, c)) == CL_SUCCESS);
    assert(kernel.setArg(1, cl::getBuffer(context, a)) == CL_SUCCESS);
    assert(kernel.setArg(2, cl::getBuffer(context, b)) == CL_SUCCESS);
    assert(kernel.setArg(3, vecSize) == CL_SUCCESS);

    cl::NDRange offset(0);
    cl::NDRange local(local_size);
    cl::NDRange global(global_size);

    assert(command_queue.enqueueNDRangeKernel(kernel, offset, global, local) == CL_SUCCESS);

    assert(command_queue.finish() == CL_SUCCESS);

    getTime(&t);
    printTime(&s, &t, "Run: ", "\n");


    getTime(&s);
    float error = 0.f;
    float check = 0.f;
    for(unsigned i = 0; i < vecSize; i++) {
        error += c[i] - (a[i] + b[i]);
        check += c[i];
    }
    getTime(&t);
    printTime(&s, &t, "Check: ", "\n");
    fprintf(stderr, "Error: %f\n", error);

    if (sum != check) {
        printf("Sum: %f vs %f\n", sum, check);
        abort();
    }

    cl::free(context, a);
    cl::free(context, b);
    cl::free(context, c);
    return 0;
}
