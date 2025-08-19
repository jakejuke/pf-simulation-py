# Translates Anton Manin's Matlab code to Pyhton

import os
import numpy as np
import math
# import matplotlib.pyplot as plt
# import time

def main():
    print('code to generate sim pars')
    print('use as ...')
    print('moelans(output_dir, path_anton)')

# # some testing variables
# tmp_dir = '/var/folders/hs/1nx30dpj1_v6db9515345nxr0000gn/T/pfsim_2ckzehw8'
# path_anton = os.path.dirname(__file__)
# # Generate arrays for GB energy (sigma) and GB mobility (mu)
# sigma = generate_em('const', 0.25)
# mu = generate_em('const', 0.55302)

def moelans(sigma: np.ndarray, mu: np.ndarray, output_dir: str, path_anton: str):
    
    # print alive msg at beginning of main()
    alive()
    
    GAMMA, G_GAMMA, ETA = load_tables(path_anton)
    
    # Anton defines these values in his moelans.m code
    # There are also lists of other values that he tested
    l_gb = 4/2
    sigma_init = 0.25
    gamma_init = 1.5
    
    # initial step
    g_init = f_g(gamma_init, GAMMA, G_GAMMA)
    f0 = f_f0(gamma_init, GAMMA, ETA)
    a_init = math.sqrt(f0)/g_init
    
    m = sigma_init / (l_gb * g_init * math.sqrt(f0))
    kappa = sigma_init**2 / (g_init**2 * m)
    gamma = []
    L = []
    l = []

    for i in range(6800):
        a_t = a_init
        count = 0
        while True:
            count += 1
            kappa_t = kappa
            g_t = sigma[i] / math.sqrt(kappa_t * m)
            [flag, gamma_t] = f_gamma(g_t, GAMMA, G_GAMMA)

            if flag == 1:
                print(f'i = {i}, count = {count}')

            f0_t = f_f0(gamma_t, GAMMA, ETA)
            a_t = math.sqrt(f0_t) / g_t

            # if difference is smaller than 0.1, equal!
            # 0.1 is working for energy 0.25 cutoff 20
            # 0.3 is working for energy 0.25 cutoff 10
            if abs(a_init - a_t) > 0.3:
                a_init = a_t
            else:
                break

        gamma.append(gamma_t)
        a = a_t
        L.append(mu[i] * f_g(gamma_t, GAMMA, G_GAMMA) * math.sqrt(m/kappa_t))
        l.append(math.sqrt(kappa_t / (m * f_f0(gamma_t, GAMMA, ETA))))

        # time.sleep(0.1)
        # print(f"i = {i}, count = {count}, "
        #       f"gamma[i] = {gamma[i]}, L[i] = {L[i]}, l[i] = {l[i]}")

    ## write numpy arrays generated above to txt files
    # probably don't need these (Anton's old code generated them for his
    # moelans.m function)
    np.savetxt(os.path.join(output_dir, 'sigma_gb.txt'), sigma, fmt='%1.6f')
    np.savetxt(os.path.join(output_dir, 'mobility_gb.txt'), mu, fmt='%1.6f')
    # write some single values to a txt file
    var_to_txt(os.path.join(output_dir, 'm.txt'), m)
    var_to_txt(os.path.join(output_dir, 'k.txt'), kappa)
    # gamma is always(?) 1.5, but it was like that for Anton's code too
    list_to_txt(os.path.join(output_dir, 'gamma.txt'), gamma)
    list_to_txt(os.path.join(output_dir, 'L.txt'), L)
    list_to_txt(os.path.join(output_dir, 'lgb.txt'), l)

def alive():
    print("\nNow generating parameter files for phase-field simulation ...\n")

def generate_em(m_type: str, m_val: float, x = [0, 0.01, 68],
                      par_list = None) -> np.ndarray:
    """Generates the energy/mobility array to be used by moelans()

    Args:
        m_type (str): type of mobility function (e.g. const, step, ...)
        m_val (float): mobility value(s)
        x (list, optional): x values for which to generage corresponding
            mobility values. Defaults to [0, 0.01, 68].
        par_list (_type_, optional): Maybe a better way to input
            additional parameters in the future. Defaults to None.

    Returns:
        np.ndarray: values of mobility over range of x
    
    Notes:
        Originally from Anton or Mingyan (generate_mobility.m)
    """
    # default values [0, 0.01, 68] were ones used by Anton
    nof_x = math.ceil((x[-1] - x[0])/x[1])
    
    if m_type.lower() == "const":
        return np.ones(nof_x) * m_val
    else:
        # How can I add proper error handling / traceback?
        #raise SystemExit(1)
        raise Exception("only works for constant mobility functions so far...")

def f_g(gamma: float, GAMMA, G_GAMMA):
    if gamma < 0.5:
        raise Exception("gamma should be larger than 0.5")
    else:
        i = np.argmin(np.abs(GAMMA-gamma))
        d = GAMMA[i] - gamma
        #plt.plot(np.abs(GAMMA-gamma))
        #plt.ylabel('some numbers')
        #plt.show()
        
        if d > 0:
            delta_x = GAMMA[i] - GAMMA[i-1]
            delta_y = G_GAMMA[i] - G_GAMMA[i-1]
            m = delta_y/delta_x
            c = G_GAMMA[i] - m*GAMMA[i]
            return m*(GAMMA[i] - np.abs(d)) + c
        elif d < 0:
            delta_x = GAMMA[i+1] - GAMMA[i]
            delta_y = G_GAMMA[i+1] - G_GAMMA[i]
            m = delta_y/delta_x
            c = G_GAMMA[i] - m*GAMMA[i]
            return m*(GAMMA[i] + np.abs(d)) + c
        else:
            return G_GAMMA(i)

def f_f0(gamma: float, GAMMA, ETA):
    if gamma < 0.5:
        raise Exception("gamma should be larger than 0.5")
    else:
        i = np.argmin(np.abs(GAMMA-gamma))
        d = GAMMA[i] - gamma
        #plt.plot(np.abs(GAMMA-gamma))
        #plt.ylabel('some numbers')
        #plt.show()
        
        if d > 0:
            delta_x = GAMMA[i] - GAMMA[i-1]
            delta_y = ETA[i] - ETA[i-1]
            m = delta_y/delta_x
            c = ETA[i] - m*GAMMA[i]
            eta_inter = m*(GAMMA[i] - np.abs(d)) + c
        elif d < 0:
            delta_x = GAMMA[i+1] - GAMMA[i]
            delta_y = ETA[i+1] - ETA[i]
            m = delta_y/delta_x
            c = ETA[i] - m*GAMMA[i]
            eta_inter = m*(GAMMA[i] + np.abs(d)) + c
        else:
            eta_inter = ETA(i)
        
        # I have no idea where this equation comes from and why it is 
        # written like this!
        return (2*(0.25*eta_inter**4 - 0.5*eta_inter**2)
                + gamma * eta_inter**2 * eta_inter**2 + 0.25)

# According to Mingyan's matlab code, this function is very important
def f_gamma(g: float, GAMMA, G_GAMMA):
    i = np.argmin(np.abs(G_GAMMA-g))
    d = G_GAMMA[i] - g
    #plt.plot(np.abs(GAMMA-gamma))
    #plt.ylabel('some numbers')
    #plt.show()
    
    if d > 0:
        delta_x = G_GAMMA[i] - G_GAMMA[i-1]
        delta_y = GAMMA[i] - GAMMA[i-1]
        m = delta_y/delta_x
        c = GAMMA[i] - m*G_GAMMA[i]
        res = m*(G_GAMMA[i] - np.abs(d)) + c
    elif d < 0:
        delta_x = G_GAMMA[i+1] - G_GAMMA[i]
        delta_y = GAMMA[i+1] - GAMMA[i]
        m = delta_y/delta_x
        c = GAMMA[i] - m*G_GAMMA[i]
        res = m*(G_GAMMA[i] + np.abs(d)) + c
    else:
        res = GAMMA(i)
    
    # calculated gamma shoule not be smaller than 0.5
    # assign 0.53 to these values
    # value smaller than 0.53 will be in a endless loop
    if res < 0.5:
        print("Warning! calculated gamma should NOT be smaller than"
              "0.5, assign 0.53")
        flag = 1
        res = 0.53
        return flag, res
    else:
        flag = 0
        return flag, res

def load_tables(path: str):
    """
    Loads csv files
    Anton had three .mat files that he loaded
    I guess these were some values from literature
    """
    with open(os.path.join(path, "GAMMA.csv"), "r") as f:
        GAMMA = np.loadtxt(f)
    #
    with open(os.path.join(path, "g_gamma.csv"), "r") as f:
        G_GAMMA = np.loadtxt(f)
    #
    with open(os.path.join(path, "eta_inter.csv"), "r") as f:
        ETA = np.loadtxt(f)
        
    return GAMMA, G_GAMMA, ETA

def list_to_txt(txt_file: str, L: list):
    with open(txt_file, 'w') as f:
        for item in L:
            # both lines do the same (default is 6)
            # line = '{:.6f}\n'.format(item)
            # line = '{:f}\n'.format(item)
            # 'g' is general format
            line = '{:g}\n'.format(item)
            f.write(line)

def var_to_txt(txt_file: str, v):
    with open(txt_file, 'w') as f:
        # 'g' is general format
        line = '{:g}\n'.format(v)
        f.write(line)

if __name__ == "__main__":
    main()


""" 
GAMMA, G_GAMMA = load_gammas()
print("Loading GAMMA")
print(GAMMA[0], GAMMA[1])
print("Loading G_GAMMA")
print(G_GAMMA[0], G_GAMMA[1])

if isinstance(GAMMA, np.ndarray):
    print("GAMMA is a numpy array")
else:
    print("GAMMA is not a numpy array")

# Anton used 6800 values
step_deg = 0.01
x_deg = np.arange(0 + step_deg, 68 + step_deg, step_deg)

print(f"mu : mu shape = {mu.shape}, mu data type = {mu.dtype}")
fig, ax = plt.subplots()
ax.plot(x_deg, mu)
ax.set(xlabel='misor', ylabel='mobility (a.u.)',
    title='Mobility Input')
ax.grid()
# fig.savefig("test.png")
plt.show()
"""