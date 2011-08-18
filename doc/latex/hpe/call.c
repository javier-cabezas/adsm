    ecl_kernel kernel;
    ecl_error error_code;
    cl_mem mem;
    cl_uint global_size = vec_size;

    error_code = eclGetKernel("vecAdd", &kernel);
    if(error_code != eclSuccess) return error(error_code);

    error_code = eclSetKernelArgPtr(&kernel, 0, c);
    if(mem == NULL || error_code != eclSuccess)
        return error(error_code);
    error_code = eclSetKernelArgPtr(&kernel, 1, a);
    if(mem == NULL || error_code != eclSuccess)
        return error(error_code);
    error_code = eclSetKernelArgPtr(&kernel, 2, b);
    if(mem == NULL || error_code != eclSuccess)
        return error(error_code);

    error_code = eclCallNDRange(&kernel, 1, NULL,
                                &global_size, NULL);
    if(error_code != eclSuccess)  return error(error_code);

