#ifndef AUX_CHECKFILEOP_HPP
#define AUX_CHECKFILEOP_HPP

#include <cstdio>

#define CHECK_FILE_READ(opname, filename, ...) \
	{ \
		auto res = opname(filename, ##__VA_ARGS__); \
		if (res==-1 || res==0) { \
			std::printf("file loading operation at line (%d); ",__LINE__); \
			std::printf("in file %s\n", __FILE__); \
			std::printf("\tfor %s", filename); \
			std::printf("\treturned %d\n\taborting...\n", res); \
			return (2); \
		} else { \
			std::printf("%d lines/elements read from %s\n", res, filename); \
		} \
	}
	
#define CHECK_FILE_WRITE(opname, filename, ...) \
	{ \
		auto res = opname(filename, ##__VA_ARGS__); \
		if (res==-1 || res==0) { \
			std::printf("file writing operation at line (%d) ",__LINE__); \
			std::printf("in file %s\n", __FILE__); \
			std::printf("\tfor %s", filename); \
			std::printf("\treturned %d\n\taborting...\n", res); \
			return (3); \
		} else { \
			std::printf("%d lines/elements stored in %s\n", res, filename); \
		} \
	}
	
#define CHECK_FILE_COPY(opname, filename, ...) \
	{ \
		auto res = opname(filename, ##__VA_ARGS__); \
		if (res == false) { \
			std::printf("file copy operation at line (%d) ",__LINE__); \
			std::printf("in file %s\n", __FILE__); \
			std::printf("\tfor %s", filename); \
			std::printf("\treturned %d\n\taborting...\n", res); \
			return (3); \
		} else { \
			std::printf("%d lines/elements copied to %s\n", res, filename); \
		} \
	}

#endif // AUX_CHECKFILEOP_HPP
