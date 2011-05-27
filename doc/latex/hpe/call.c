    ocl_kernel kernel;
    ocl_error error_code;
    cl_mem mem;
    cl_uint global_size = vec_size;

    error_code = oclGetKernel("vecAdd", &kernel);
    if(error_code != oclSuccess) return error(error_code);

    mem = cl_mem(oclPtr(c));
    error_code = oclSetKernelArg(&kernel, 0, sizeof(mem), &mem);
    if(mem == NULL || error_code != oclSuccess)
        return error(error_code);                       
    mem = cl_mem(oclPtr(a));                            
    error_code = oclSetKernelArg(&kernel, 1, sizeof(mem), &mem);
    if(mem == NULL || error_code != oclSuccess)
        return error(error_code);                       
    mem = cl_mem(oclPtr(b));                            
    error_code = oclSetKernelArg(&kernel, 2, sizeof(mem), &mem);
    if(mem == NULL || error_code != oclSuccess)
        return error(error_code);

    error_code = oclCallNDRange(&kernel, 1, NULL,
                                &global_size, NULL);
    if(error_code != oclSuccess)  return error(error_code);

