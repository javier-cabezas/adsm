
extern "C"
__global__
void inc(float *A);

__global__
void inc(float *A)
{
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

    A[idx] += float(1);
}
