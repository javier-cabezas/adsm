    ocl_kernel kernel;
    ocl_error error_code;
    cl_mem mem;
    cl_uint global_size = vec_size;

    error_code = __oclKernelGet("vecAdd", &kernel);
    if(error_code != oclSuccess) return error(error_code);

    error_code = __oclKernelConfigure(&kernel, 1, NULL,
                                      &global_size, NULL);
    if(error_code != oclSuccess) return error(error_code);

    mem = cl_mem(oclPtr(c));
    error_code = __oclKernelSetArg(&kernel, &mem, sizeof(mem), 0);
    if(mem == NULL || error_code != oclSuccess)
        return error(error_code);
    mem = cl_mem(oclPtr(a));
    error_code = __oclKernelSetArg(&kernel, &mem, sizeof(mem), 1);
    if(mem == NULL || error_code != oclSuccess)
        return error(error_code);
    mem = cl_mem(oclPtr(b));
    error_code = __oclKernelSetArg(&kernel, &mem, sizeof(mem), 2);
    if(mem == NULL || error_code != oclSuccess)
        return error(error_code);

    error_code = __oclKernelLaunch(&kernel);
    if(error_code != oclSuccess)  return error(error_code);

