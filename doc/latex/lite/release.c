    FILE *fp;

    /* Release input vectors */
    clFree(lite.context[0], a);
    clFree(lite.context[0], b);

    /* Write and release the output vector */
    fp = fopen(vector_c_file, "w");
    if(fp == NULL) {
        fprintf(stderr, "Cannot write output %s\n",
                vector_c_file);
        clFree(lite.context[0], c); abort();
    }
    fwrite(c, vector_size, sizeof(float), fp);
    fclose(fp);
    clFree(lite.context[0], c);
    clLiteRelease(lite);
