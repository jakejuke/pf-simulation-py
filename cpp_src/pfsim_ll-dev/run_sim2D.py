# Code to automatically run C++ simulations

import os
import sys
import shutil
import re
import h5py
import numpy as np
import tempfile
import argparse
from datetime import datetime
from glob import glob
from natsort import natsorted

# import my packages
path_anton = os.path.expanduser('~/Code/pfsim_ll/anton')
sys.path.append(path_anton)
import anton
import prep_sim2D

## User defined constants
# could add path to sim code as input argument ...
PFSIM_DIR = os.path.expanduser('~/Code/pfsim_ll')
H5_G_SIM = 'sim_data' # h5 group name for simulation pars and output
OUTPUT_PATTERN = 'lm_partition_*' # prefix of sim output files

# required keys in h5 input file
required_keys = {'group_i': 'input',
                 'ds_map': 'id_map',
                 'ds_ori': 'orientations',
                 'attr_e': 'energy',
                 'attr_m': 'mobility',
                 'attr_stps': 'sim_steps',
                 'attr_intv': 'save_interval'}

def run_sim2D(h5_fin: str):

    # create temp directory to save all par files and sim output
    tmp_dir = tempfile.mkdtemp(prefix='pfsim_')
    print('created temporary directory:\n', tmp_dir)
    
    # get map, orientations and input parameters from h5 file
    print('loading data from h5 file')
    id_map, orients, input_pars = read_h5_file(h5_fin, required_keys)
    [M, N] = id_map.shape # dims of id_map (rows = M, columns = N)
    num_grains = id_map.max()
    mobility = input_pars[required_keys['attr_m']]
    energy = input_pars[required_keys['attr_e']]

    # now write ops and orimap txt files to temp dir
    print('\ncreating input files for C++ code')
    f_ops = write_ops(id_map, tmp_dir)
    f_orimap = write_orimap(id_map, orients, tmp_dir)
    print(f_ops)
    print(f_orimap)

    # Update h5 file
    # writes new h5 file to tmp_dir for now
    h5_fout = os.path.join(tmp_dir, 'temp.h5')
    shutil.copyfile(h5_fin, h5_fout)
    # add more sim parameters to h5 file
    sim_pars = {'sim_steps': input_pars[required_keys['attr_stps']],
                'save_interval': input_pars[required_keys['attr_intv']],
                'map_dims': [M, N],
                'num_grains': num_grains,
                'output_dir': tmp_dir,
                'pfsim_dir': PFSIM_DIR}
    update_h5_4_sim(h5_fout, H5_G_SIM, sim_pars)

    ## Generate data for GB energy (sigma) and GB mobility (mu)
    # and run moelans() to write txt files (L, gamma, k, m)
    sigma = anton.generate_em('const', energy)
    if np.size(mobility) == 6800:
        mu = mobility
        print('custom mobilities from h5 input (6800 values)')
    else:
        mu = anton.generate_em('const', mobility)
    anton.moelans(sigma, mu, tmp_dir, path_anton)

    # prep_sim2D updates C++ files (sim2d.cpp, config2d.hpp, Makefile)
    prep_sim2D.prep_sim(h5_fout)

    ## Run Sim
    print('running phase-field simulation ...')
    # _ _ should use subprocess.call() instead of os.system() _ _
    cwd = os.getcwd()
    os.chdir(os.path.join(PFSIM_DIR, 'exec'))
    # os.system('terminal command')
    os.system('make -B misorientation')
    os.system('make -B sim2d')
    os.system('./sim2d')
    # get misorientations used in simulation from binary file
    lin_misors = np.fromfile('misorientation', dtype=np.float32)
    os.chdir(cwd)

    ## Write output to h5 and clean up
    # This function appends the temp h5 file created earlier
    # Could this lead to fragmentation? Repack, maybe?
    print('writing output to .h5 file')
    sim_post(tmp_dir, OUTPUT_PATTERN, h5_fout, H5_G_SIM, cmpr='gzip')

    # make np (lower triangle) array with misorientations and save as data set
    misor_mat = np.zeros((num_grains, num_grains))
    misor_mat[np.tril_indices(num_grains, -1)] = lin_misors
    add_h5_dset(h5_fout, H5_G_SIM, 'misorientations', misor_mat)
    
    # basename of h5 file without .h5 extension
    h5_base = h5_fin.rsplit('.', maxsplit=1)[0]
    shutil.copyfile(h5_fout, h5_base + '_out.h5')

    # rm tmp_dir
    print('removing temp output')
    shutil.rmtree(tmp_dir)
    print('all done')


def read_h5_file(h5_file: str, required: dict):
    
    # dict for input parameters
    input_pars = {}
    with h5py.File(h5_file, 'r') as h5f:
        g_input = h5f[required['group_i']]
        # get datasets from group "input"
        id_map = np.array(g_input[required['ds_map']])
        orients = list(g_input[required['ds_ori']])
        # get input parameters from attributes
        for item in required:
            if item.startswith('attr_'):
                print(item, ':', required[item])
                input_pars[required[item]] = g_input.attrs[required[item]]

    return id_map, orients, input_pars

# write_h5_file NOT used right now
def write_h5_file(h5_fout, id_map, orientations, energy, mobility,
                  sim_steps, save_interval):
    
    now = datetime.today().isoformat()

    with h5py.File(h5_fout, 'w') as h5f:
        # create group "input" and save grain id map and orientations
        g_input = h5f.create_group('input')
        g_input.create_dataset('id_map', data=id_map, compression='gzip')
        g_input.create_dataset('orientations', data=orientations, compression='gzip')
        # save input parameters for the simulation as group attributes
        g_input.attrs['date'] = now
        g_input.attrs['energy'] = energy
        g_input.attrs['energy_func'] = 'const'
        g_input.attrs['mobility'] = mobility
        g_input.attrs['mobility_func'] = 'custom'
        g_input.attrs['sim_steps'] = sim_steps
        g_input.attrs['save_interval'] = save_interval

def update_h5_4_sim(h5_file_in: str, group: str, sim_pars: dict):

    with h5py.File(h5_file_in, 'r+') as h5f:
        # add attributes to group
        if group in h5f:
            for item in sim_pars:
                h5f[group].attrs[item] = sim_pars[item]
        else:
            print('creating new h5 group:', group)
            h5f.create_group(group)
            for item in sim_pars:
                h5f[group].attrs[item] = sim_pars[item]

def add_h5_attr(h5_file_in: str, group: str, attr_name: str, attr_val):

    with h5py.File(h5_file_in, 'a') as h5f:
        if group in h5f:
            # print('adding', data_set, 'to', data_group)
            h5f[group].attrs[attr_name] = attr_val
        else:
            print('creating new h5 group:', group)
            h5f.create_group(group)
            h5f[group].attrs[attr_name] = attr_val

def add_h5_dset(h5_file_in: str, group: str, ds_name: str, ds_val, cmpr: str = None):

    with h5py.File(h5_file_in, 'a') as h5f:
        if group in h5f:
            h5f[group].create_dataset(ds_name, data=ds_val, compression=cmpr)
        else:
            print('creating new h5 group:', group)
            h5f.create_group(group)
            h5f[group].create_dataset(ds_name, data=ds_val, compression=cmpr)

def write_orimap(a: np.ndarray, o_map: list, tmp_path: str):
    
    # ADD check if numRows = maxID
    if len(o_map) != a.max():
        raise Exception("Mismatch number of grains")
    
    # full file name
    ff_name = os.path.join(tmp_path, 'orimap.txt')
    # save with 6 decimal places
    np.savetxt(ff_name, o_map, fmt='%1.6f')
    
    return ff_name

def write_ops(a: np.ndarray, tmp_path: str):
    
    if a.ndim != 2:
        raise Exception("only for 2D arrays so far...")
    
    # full file name
    ff_name = os.path.join(tmp_path, 'map.ops')
    
    with open(ff_name, 'w') as f:
        # a simple nditer would probably work: e.g.,
        # for n in np.nditer(a, order='C')
        # but using ndindex just to be safe
        for i, j in np.ndindex(a.shape):
            #            x
            #   ---------------->
            #   | (0,0) (0,1) (0, 2) ...
            #   | (1,0) (1,1)
            # y | (2,0)
            #   | ...
            #   v
            # 
            # thus, (i,j) -> (y,x)
            # typically, python is rowMajor, which means col_index
            # changes the fastest
            x, y = (j + 1), (i + 1)
            # write text file with 4 columns: x, y, intensity, id
            line = f'{x} {y} 1 {a[i, j]}\n'
            f.write(line)
    
    # return file name to .ops file
    return f.name

def read_ops(f_ops: str, dtype=int, delimiter=None, skiprows=0):
    
    a = np.loadtxt(f_ops, dtype=dtype, delimiter=delimiter, skiprows=skiprows)
    num_rows = np.max(a[:,1])
    num_columns = np.max(a[:,0])
    a_lin = a[:,-1]
    return a_lin.reshape(num_rows, num_columns)

def sim_post(sim_dir: str, ops_prefix: str,
             h5_file: str, data_group: str, cmpr: str = None):

    # output files from simulation in naturally sorted order
    # __ Note: I don't see the order when viewing the h5 file __
    sim_output_files = natsorted(glob(os.path.join(sim_dir, ops_prefix)))

    with h5py.File(h5_file, 'r+') as h5f:
        for f_ops in sim_output_files:
            a = read_ops(f_ops)
            bname_simstep = os.path.basename(f_ops)
            step_num = re.search(r'\d+$', bname_simstep).group()
            # I wish the step_num's all had the same number of digits...
            s = 'sim_step_' + step_num
            h5f[data_group].create_dataset(s, data=a, compression=cmpr)

if __name__ == "__main__":
    
    ## Parse input args so user can run from the command line
    #  with input (i.e., str to h5 file)
    parser = argparse.ArgumentParser()
    # parser.add_argument("-f", "--file", default="admin")
    parser.add_argument("-f", "--file", default=None, help="full path to .h5")

    # gather all extra args into a list named "args.extra"
    parser.add_argument("extra", nargs='*')
    args = parser.parse_args()

    # set args.file to first positional arg if len(args.extra) == 0
    if args.file == None and len(args.extra) == 1:
        args.file = args.extra.pop(0)
    else:
        raise Exception("Check input args ...")

    print('Now running ...')
    print(f'    run_sim2D({args.file})')
    
    run_sim2D(args.file)
