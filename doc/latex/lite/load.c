    cl_lite lite;
    cl_program program;
    cl_int error_code;

    /* Initialize OpenCL */
    if(oclLiteInit(&lite) != CL_SUCCESS) return -1;
    if(lite.num_devices == 0) return -1;

    /* Load and compile the OpenCL kernel */
    program = clLiteLoad(lite, kernel_file, &error_code);
    if(error_code != CL_SUCCESS) {
        clLiteRelease(lite);
        return -1;
    }
