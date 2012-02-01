#include <fstream>

#include "file.h"

namespace __impl { namespace util {

std::string get_file_contents(const std::string &path, gmacError_t &err)
{
    std::string ret;
    err = gmacSuccess;

    std::ifstream in(path.c_str(), std::ios_base::in);
    if (!in.good()) {
        err = gmacErrorInvalidValue;
        return ret;
    }
    in.seekg (0, std::ios::end);
    std::streampos length = in.tellg();
    in.seekg (0, std::ios::beg);
    if (length == std::streampos(0)) {
        err = gmacErrorInvalidValue;
        return ret;
    }
    // Allocate memory for the code
    char *buffer = new char[int(length)+1];
    // Read data as a block
    in.read(&buffer[0], length);
    in.close();

    buffer[static_cast<int>(length)] = '\0';

    ret = std::string(&buffer[0]);

    delete [] buffer;

    return ret;
}

}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
