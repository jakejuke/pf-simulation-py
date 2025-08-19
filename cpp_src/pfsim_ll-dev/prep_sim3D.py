# Prepare everything to run a 3D simulation
# Adapted from prep_sim2D.py for Leonard Lauber

import os
import h5py
import shutil
import re
import argparse

# This code builds a dict called sim_pars from the provided h5 file. The 
# h5 file should have a group called 'sim_data' with all of these
# attributes:
# 
# sim_pars = {'sim_steps': input_pars[required_keys['attr_stps']],
#             'save_interval': input_pars[required_keys['attr_intv']],
#             'map_dims': [M, N],
#             'output_dir': tmp_dir,
#             'pfsim_dir': PFSIM_DIR}

def prep_sim(h5f: str):
    """
    h5f should have a group called sim_data with attributes:
    - map_dims, num_grains, save_interval, sim_steps
    - output_dir, pfsim_dir
    """

    ## Get sim parameters and paths from h5 file
    sim_pars = get_simpars(h5f, 'sim_data')
    num_iter = int(sim_pars['sim_steps'])
    # step_size = int(sim_pars['save_interval'])
    save_intvls = sim_pars['save_interval']
    cell_size_m = sim_pars['map_dims'][0]
    cell_size_n = sim_pars['map_dims'][1]
    cell_size_l = sim_pars['map_dims'][2]
    nof_grains = sim_pars['num_grains']
    path_cppSim = sim_pars['pfsim_dir']
    path_output = sim_pars['output_dir']

    ## Set paths and files wrt C++ code structure
    dir_exec = os.path.join(path_cppSim, 'exec')
    my_cpp = 'sim3di.cpp'
    my_config = 'config3d.hpp'
    print('source code:', my_cpp, 'at', dir_exec)

    ## Set some file paths
    f_ops = os.path.join(path_output, 'map.ops')
    f_orimap = os.path.join(path_output, 'orimap.txt')
    f_L = os.path.join(path_output, 'L.txt')
    f_gamma = os.path.join(path_output, 'gamma.txt')

    ## Get max value of L, k (kappa), gamma, m
    L_max = get_max(f_L)
    gamma_max = get_max(f_gamma)
    k_max = get_max(os.path.join(path_output, 'k.txt'))
    m_max = get_max(os.path.join(path_output, 'm.txt'))
    
    # Convert the new values to a string format that matches C++ syntax
    save_intvls_str = ', '.join(map(str, save_intvls))

    ## Update sim2d.cpp, config2d.hpp, Makefile 
    # dict with regular expression patterns and replacements
    patterns_cpp = {
        r'#define NUM_ITER \d+': f"#define NUM_ITER {str(num_iter)}",
        # r'#define STEP_SIZE \d+': f"#define STEP_SIZE {str(step_size)}",
        r'const char outdirname': f"\tconst char outdirname[128] = \"{os.path.join(path_output, '')}\";",
        r'const std::size_t m = \d+': f"\tconst std::size_t m = {str(cell_size_m)};",
        r'const std::size_t n = \d+': f"\tconst std::size_t n = {str(cell_size_n)};",
        r'const std::size_t k = \d+': f"\tconst std::size_t k = {str(cell_size_l)};",
        r'char infilename': f"\tchar infilename[128] = \"{f_ops}\";",
        r'std::vector<int> save_intervals = \{.*?\};': f"\tstd::vector<int> save_intervals = {{{save_intvls_str}}};"
    }
    # Read and write to the same line
    # Replace whole lines when the pattern is matched
    with open(os.path.join(dir_exec, my_cpp), 'r+') as f:
        lines = f.readlines()  # Read all lines into a list
        
        for i, line in enumerate(lines):
            for pattern, replacement in patterns_cpp.items():
                match = re.search(pattern, line)  # Search for the pattern in the line
                if match:
                    # Replace the matched text with the replacement text
                    lines[i] = replacement + '\n'
                    break  # Move to the next line after the first match is found
        
        # Write all modified lines back to the file
        f.seek(0) # Start at beginning of the file
        f.writelines(lines)

    # More updates for sim2d.cpp
    # Replaces the *next* line of text after matching the patterns
    more_patterns_cpp = {
        r'// files for L and gamma': f"\tCHECK_FILE_READ(sim::readcoeffmaps, \"{f_L}\", coeffmapsbuffer.data, inccoeff);",
        r'// now repeat for gamma': f"\tCHECK_FILE_READ(sim::readcoeffmaps, \"{f_gamma}\", coeffmapsbuffer.data+1, inccoeff);"
    }
    with open(os.path.join(dir_exec, my_cpp), 'r+') as f:
        lines = f.readlines()  # Read all lines into a list
        
        for i, line in enumerate(lines):
            for pattern, replacement in more_patterns_cpp.items():
                match = re.search(pattern, line)  # Search for the pattern in the line
                
                if match:
                    # Replace the matched text with the replacement text
                    lines[i+1] = replacement + '\n'
                    break  # Move to the next line after the first match is found

        # Write all modified lines back to the file
        f.seek(0) # Start at beginning of the file
        f.writelines(lines)

    # Updates to config3d.hpp
    # There are many more parameters set in the config3d.hpp file, but I
    # am only varying these ones for the ML training right now
    patterns_config = {
        r'#define NOF_GRAINS \d+': f"#define NOF_GRAINS {nof_grains}",
        r'#define LCOEFF \d+': f"#define LCOEFF {L_max}",
        r'#define KAPPACOEFF \d+': f"#define KAPPACOEFF {k_max}",
        r'#define GAMMACOEFF \d+': f"#define GAMMACOEFF {gamma_max}",
        r'#define MCOEFF \d+': f"#define MCOEFF {m_max}"
    }
    with open(os.path.join(dir_exec, my_config), 'r+') as f:
        lines = f.readlines()

        for i, line in enumerate(lines):
            for pattern, replacement in patterns_config.items():
                match = re.search(pattern, line)  # Search for the pattern in the line
                
                if match:
                    # Replace the matched text with the replacement text
                    lines[i] = replacement + '\n'
                    break  # Move to the next line after the first match is found
        
        # Write all modified lines back to the file
        f.seek(0) # Start at beginning of the file
        f.writelines(lines)

    # Updates to Makefile
    # Auto write new NOF_GRAINS in this line:
    #   $(CXX) $(CFLAGS) -DNOF_GRAINS=157 -o $@ $^
    with open(os.path.join(dir_exec, 'Makefile'), 'r+') as f:
        lines = f.readlines()
        keyword = '-DNOF_GRAINS'
        
        for i, line in enumerate(lines):
            if keyword in line:
                # Split the line into words using whitespace as the separator
                words = line.split()
                
                for j, word in enumerate(words):
                    match = re.search(keyword, word)
                    if match:
                        words[j] = '{}={}'.format(keyword, nof_grains)
                        break
                
                # Write the modified line back to the file
                lines[i] = '\t' + ' '.join(words) + '\n'
        
        # Write all modified lines back to the file
        f.seek(0)
        f.writelines(lines)

    # Updates to generate_misorientation.cpp
    # (No updates needed currently)

    ## Copy orimap to exec folder
    shutil.copyfile(f_orimap, os.path.join(dir_exec, 'orimap'))

def get_max(txt_file: str) -> float:
    with open(txt_file, 'r') as f:
        vals = [line.rstrip() for line in f]
        return max(vals)

def get_simpars(h5_file: str, group: str) -> dict:

    d = {}
    with h5py.File(h5_file, 'r') as h5f:
        # Open group 'sim_pars'
        sim_data = h5f[group]
        # d['num_iter'] = sim_settings['num_iter'][()]
        for key in sim_data.attrs.keys():
            d[key] = sim_data.attrs[key]
    
    return d


if __name__ == '__main__':

    ## This code allows the user to run prep_sim from the command line
    #  with input (i.e., str to h5 file)

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", default=None)

    # gather all extra args into a list named "args.extra"
    parser.add_argument("extra", nargs='*')
    args = parser.parse_args()

    # set args.file to first positional arg if len(args.extra) == 0
    if args.file == None and len(args.extra) == 1:
        args.file = args.extra.pop(0)
    else:
        raise Exception("Problem with input arguments")

    print('Now running ...')
    print(f'    prep_sim2D({args.file})')
    
    prep_sim(args.file)
