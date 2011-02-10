SECTIONS {
    .rodata.ocl_code : {
        __ocl_code_start = .;
        *(.ocl_code .ocl_code.*)
        __ocl_code_end = .;
    }
}
