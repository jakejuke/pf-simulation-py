#ifndef SRC_AUX_COPYFILE_HPP
#define SRC_AUX_COPYFILE_HPP 1

#include <fstream>

namespace aux {
// function needed to provide c++0x support;
// this implementation does not include any error handling;
// in newer standards use std::filesystem::copy_file instead;
// copy in binary mode;
bool copy_file(const char *srcfilename, const char* destfilename) {
    std::ifstream src(srcfilename, std::ios::binary);
    std::ofstream dest(destfilename, std::ios::binary);
    dest << src.rdbuf();
    return src.good() && dest.good();
}

} /* namespace aux */

#endif // SRC_AUX_COPYFILE_HPP
