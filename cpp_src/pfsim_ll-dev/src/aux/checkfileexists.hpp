#ifndef SRC_AUX_CHECKFILEEXISTS_HPP
#define SRC_AUX_CHECKFILEEXISTS_HPP

namespace aux {

inline bool checkfileexists(const char filename[]) {
	if (FILE *fil = fopen(filename, "r")) {
		fclose(fil);
		return true;
	} else {
		return false;
	}
}

} /* namespace aux */

#endif // SRC_AUX_CHECKFILEEXISTS_HPP
