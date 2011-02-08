SECTIONS {
    .rodata.gmac_code : {
        __gmac_code_start = .;
        *(.gmac_code .gmac_code.*)
        __gmac_code_end = .;
    }
}
