int load_vector(const char *file_name, float *vector, size_t size)
{
    FILE *fp;
    size_t n;

    fp = fopen(vector_a_file, "b");
    if(fp == NULL) return -1;
    n = fread(a, vector_size, sizeof(float), fp);
    fclose(fp);

    if(n != vector_size) return -1;
    return n;
}

. . .

    float *a, *b, *c;
    ocl_error error_code;

    /* Allocate the input and output vectors */
    error_code = clMalloc((void **)&a,
                          vector_size * sizeof(float));
    if(error_code != oclSuccess) return error(error_code);
    error_code = clMalloc((void **)&b,
                          vector_size * sizeof(float));
    if(error_code != oclSuccess) {
        clFree(a); return error(error_code);
    }
    error_code = clMalloc((void **)&c,
                          vector_size * sizeof(float));
    if(error_code != oclSuccess) {
        clFree(a); clFree(b);
        return error(error_code);
    }

    /* Initialize the input vectors */
    if(load_vector(vector_a_file, a) < 0) {
        fprintf(stderr, "Error loading %s\n", vector_a_file);
        clFree(a); clFree(b); clFree(c); abort();
    }
    if(load_vector(vector_b_file, b) < 0) {
        fprintf(stderr, "Error loading %s\n", vector_b_file);
        clFree(a); clFree(b); clFree(c); abort();
    }
