__kernel void vecAdd2(__global float *c, __global const float *a, __global const float *b, unsigned size)
{
    unsigned i = get_global_id(0);
    if(i >= size) return;

    c[i] = a[i] + b[i];
}
