    ecl_kernel kernel;
    ecl_error error_code;
    cl_mem mem;
    cl_uint global_size = vec_size;

    error_code = eclGetKernel("vecAdd", &kernel);
    if(error_code != eclSuccess) return error(error_code);

    mem = cl_mem(eclPtr(c));
    error_code = eclSetKernelArg(&kernel, 0, sizeof(mem), &mem);
    if(mem == NULL || error_code != eclSuccess)
        return error(error_code);                       
    mem = cl_mem(eclPtr(a));                            
    error_code = eclSetKernelArg(&kernel, 1, sizeof(mem), &mem);
    if(mem == NULL || error_code != eclSuccess)
        return error(error_code);                       
    mem = cl_mem(eclPtr(b));                            
    error_code = eclSetKernelArg(&kernel, 2, sizeof(mem), &mem);
    if(mem == NULL || error_code != eclSuccess)
        return error(error_code);

    error_code = eclCallNDRange(&kernel, 1, NULL,
                                &global_size, NULL);
    if(error_code != eclSuccess)  return error(error_code);

