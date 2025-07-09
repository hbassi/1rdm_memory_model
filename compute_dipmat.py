import re
import numpy as np
import argparse
import sys
import os
import shutil
import time
from gauss_hf import *

parser = argparse.ArgumentParser(prog='compute_dipmat',
                                 description='compute the CI-basis dipole moment matrix in the z direction')

parser.add_argument('--mol', required=True, help='molecule')
parser.add_argument('--basis', required=True, help='basis')
parser.add_argument('--inpath', required=False, help='custom path to log files and CI coeffs (inputs) that will be loaded')
parser.add_argument('--outpath', required=False, help='custom path to .npy files (outputs) that will be saved')

# actually parse command-line arguments
args = parser.parse_args()

mol = args.mol
basis = args.basis

# set path to log files
if args.inpath:
    path = args.inpath
else:
    path = './logfiles/'

# set path to outputs
if args.outpath:
    outpath = args.outpath
else:
    outpath = path

# construct prefix used to load and save files
if basis=='sto-3g':
    prefix='casscf22_s2_'
elif basis=='6-31g':
    prefix='casscf24_s15_'
else:
    print("Error: basis set not recognized! Must choose either sto-3g or 6-31g")
    sys.exit(1)

# check if molecule is represented
if mol!='heh+' and mol!='h2':
    print("Error: molecule not recognized! Must choose either heh+ or h2")
    sys.exit(1)

ident = prefix+mol+'_'+basis

# dipole moment conversion factor: 1 debye = 0.393430307 a.u.
DtoAU = 0.39343
AUtoD = 1./DtoAU

# energy conversion factor: 1 a.u. = 27.211396 eV
AUtoEV = 27.211396
EVtoAU = 1./AUtoEV

# define square root of 2 - bfh
rt2 = np.sqrt(2)

Logfile = path+ident+'.log'

Log = Logfile.split('/')[-1].split('.')[0]
terms = Logfile.split('/')[-1].split('_')

ci_str = list(terms[0])
# print(ci_str)
if ci_str[1] == 'a':
    ci = 'casscf'
else:
    print('not the right .LOG file for this code.')
    quit()

#specify CAS here
cas = ''.join(ci_str[6:])
sN = terms[-3]
basis = terms[-1].split('.')[0]
molecule = terms[-2]
ext = '.'+terms[-1].split('.')[1]

# read the .LOG file after copying it to 'logfile.tmp'
outp = 'logfile.tmp'
shutil.copy(Logfile, outp)

datafile = log_data(outp)
log_lines = datafile.loglines

# read parameters from the Gaussian .LOG file of CASSCF calculation
#  (currently works for restricted reference, minimum spin configuration only)
#  - KR, 05/17/2020
for (n,line) in enumerate(log_lines):
    try:
        NMOs = datafile.nao        # total number of basis fns/MOs with double occupancy
        NELECT_A = datafile.n_a    # number of alpha electrons in the system
        NELECT_B = datafile.n_b    # number of beta electrons in the system
        NOCC = NELECT_A            # total number of occupied MOs in the reference
        if ('NAtoms' in line):
            elements = log_lines[n].split()
            NAtoms = int(float(elements[1]))      # total numer of atoms in the system
        if ('NO OF BASIS FUNCTIONS' in line):
            elements = log_lines[n].split()
            NStates = int(float(elements[5]))     # number of CI states/CSFs
            print('for "CASSCF(NRoot=N)" option, please make sure N is the same as the \n total no. of configurations, NStates\n')
        if ('NO. OF ORBITALS' in line):
            elements = log_lines[n].split('=')
            NCAS = int(float(elements[1]))        # number of doubly occupied MOs in the active space
        if ('NO. OF ELECTRONS' in line):
            elements = log_lines[n].split('=')
            NELECT = int(float(elements[1]))      # total number of electrons in the active space
            if ((NELECT % 2) == 1):
                NELECT_CAS_A = int((NELECT-1)/2)
                NELECT_CAS_A += 1                 # number of alpha electrons in the active space
                NELECT_CAS_B = NELECT - NELECT_CAS_A
            else:
                NELECT_CAS_A = int(NELECT/2)
                NELECT_CAS_B = NELECT_CAS_A
            NFRZ = NELECT_A - NELECT_CAS_A        # number of doubly occupied MOs outside of the active space
    except (ValueError, IndexError, TypeError, NameError):
        print('Error encountered while reading parameters.')
        break

dipX = datafile.get_dipole_x_AO()
dipY = datafile.get_dipole_y_AO()
dipZ = datafile.get_dipole_z_AO()

# specific to CASSCF calculation
MO = np.zeros([NMOs,NMOs], np.float64)

line_num = []
for (n, line) in enumerate(log_lines):
    try:
        if ('FINAL COEFFICIENT MATRIX' in line):
            line_num.append(n)
    except (IndexError, ValueError):
        pass

count = -1
nline = line_num[count]
loops = int(NMOs / 10) + 1
last = NMOs % 10
for i in range(NMOs):
    for k in range(loops):
        try:
            if (k == (loops - 1)):
                end = last
            else:
                end = 10
            dum1 = nline+1+i*(1+loops)+(k+1)
            elements=log_lines[dum1].split()
            for j in range(end):
                s = k*10 + j
                m = i
                MO[s,m] = float(elements[j])
        except (IndexError, ValueError):
            break

dipXMO = MO.T.dot(dipX).dot(MO)
dipYMO = MO.T.dot(dipY).dot(MO)
dipZMO = MO.T.dot(dipZ).dot(MO)

# specific to CASSCF calculation
M = NStates
N = M

# specific to CASSCF calculation
#
# calculating transition dipole moments b/w CASSCF ground state
#   and excited states
#
dipXCI = np.zeros([M,M], np.float64)
dipYCI = np.zeros([M,M], np.float64)
dipZCI = np.zeros([M,M], np.float64)

densCI_AO_a = np.zeros([NMOs,NMOs], np.float64)
densCI_AO_b = np.zeros([NMOs,NMOs], np.float64)

print('ground to excited state transitions...')

last = NMOs % 5
if (last == 0):
    loops = int(NMOs / 5)
else:
    loops = int(NMOs / 5) + 1
if (NCAS % 5 == 0):
    loops_cas = int(NCAS / 5)
else:
    loops_cas = int(NCAS / 5) + 1
skip_lines = loops_cas
# print(loops,loops_cas)
for i in range(loops_cas):
    skip_lines += NCAS - 5*i
# print(skip_lines)
for (n,line) in enumerate(log_lines):
    if ('MO Ground to excited state density ' in line and '(alpha):' == line.split()[-1]):
        st1 = int(float(log_lines[n-2].split()[-1])) - 1
        st2 = int(float(log_lines[n-1].split()[-1])) - 1
        elements = log_lines[n+2]
        # read the alpha and beta transition densities in AO basis
        # AOdensline_a = n+1+2+2*loops*(NMOs+1)+2+1
        # print(n+1,log_lines[n])
        AOdensline_a = n+2*(loops_cas)*(NCAS+1)+3
        # print(AOdensline_a+1,log_lines[AOdensline_a])
        for k in range(loops):
            for i in range(NMOs):
                try:
                    if (k == (loops - 1)):
                        end = last
                    else:
                        end = 5
                    dum1 = AOdensline_a+k*(1+NMOs)+i+1
                    dum2 = dum1+1+loops*(NMOs+1)
                    elements_a = log_lines[dum1].split()
                    elements_b = log_lines[dum2].split()
                    # if st2 < 4:
                    #     print(dum1+1,elements_a)
                    #     print(dum2+1,elements_b)
                    for j in range(end):
                        s = k*5 + j
                        m = i
                        densCI_AO_a[m,s] = float(elements_a[j+1])
                        densCI_AO_b[m,s] = float(elements_b[j+1])
                except (IndexError, ValueError):
                    pass
        densCI_AO = densCI_AO_a + densCI_AO_b
        if st2 < 4:
            print(st1,st2,'\n',densCI_AO,'\n')
        dipXCI[st1,st2] = -1*np.trace(np.matmul(densCI_AO,dipX))
        dipXCI[st2,st1] = dipXCI[st1,st2]
        dipYCI[st1,st2] = -1*np.trace(np.matmul(densCI_AO,dipY))
        dipYCI[st2,st1] = dipYCI[st1,st2]
        dipZCI[st1,st2] = -1*np.trace(np.matmul(densCI_AO,dipZ))
        dipZCI[st2,st1] = dipZCI[st1,st2]

#
# calculating transition dipole moments b/w CASSCF excited states
#

print('excited to excited state transitions...')

for (n,line) in enumerate(log_lines):
    if ('Alpha transition density between states' in line):
        elements = log_lines[n].split(':')[0].split()
        st1 = int(float(elements[-1])) - 1
        st2 = int(float(elements[-2])) - 1
        densCI_AO_a = np.zeros([NMOs,NMOs], np.float64)
        densCI_AO_b = np.zeros([NMOs,NMOs], np.float64)
        for k in range(loops):
            for i in range(NMOs):
                try:
                    if (k == (loops - 1)):
                        end = last
                    else:
                        end = 5
                    dum1 = n+2+k*(1+NMOs)+i
                    dum2 = dum1+1+loops*(NMOs+1)
                    elements_a = log_lines[dum1].split()
                    elements_b = log_lines[dum2].split()
                    print(st1,st2,'\n',elements_a,'\n',elements_b)
                    for j in range(end):
                        s = k*5 + j
                        m = i
                        densCI_AO_a[m,s] = float(elements_a[j+1])
                        densCI_AO_b[m,s] = float(elements_b[j+1])
                except (IndexError, ValueError):
                    break
        densCI_AO = densCI_AO_a + densCI_AO_b
        print(st1,st2,'\n',densCI_AO/2,'\n')
        dipXCI[st1,st2] = -1*np.trace(np.matmul(densCI_AO,dipX))
        dipXCI[st2,st1] = dipXCI[st1,st2]
        dipYCI[st1,st2] = -1*np.trace(np.matmul(densCI_AO,dipY))
        dipYCI[st2,st1] = dipYCI[st1,st2]
        dipZCI[st1,st2] = -1*np.trace(np.matmul(densCI_AO,dipZ))
        dipZCI[st2,st1] = dipZCI[st1,st2]
        print(st1, st2,'\n', dipZCI[st1,st2])
        print(st1,st2,'\n',densCI_AO_a)

#
# calculating state dipole moments of all CASSCF states
#

print('stationary states...')

for (n,line) in enumerate(log_lines):
    if ('1st state is' in line):
        st1 = int(float(log_lines[n].split()[-1])) - 1
        st2 = int(float(log_lines[n+1].split()[-1])) - 1
        if (st1 == st2):
            print('state = {}'.format(st1))
            elements = log_lines[n+2]
            if ('MO valence' in log_lines[n+2]):
                densCI_AO_a = np.zeros([NMOs,NMOs], np.float64)
                shift = 0
                # read the alpha state densities in AO basis
                AOdensline_a = n+2*(loops_cas+1)+2*skip_lines
                # print(AOdensline_a,log_lines[AOdensline_a])
                for k in range(loops):
                    try:
                        irange = NMOs - k*5
                        for i in range(irange):
                            if (k == (loops - 1)):
                                if (i < last):
                                    end = i + 1
                                else:
                                    end = last
                            else:
                                if (i <= 4):
                                    end = i + 1
                                else:
                                    end = 5
                            dum1 = AOdensline_a+k+shift+i+2
                            elements=log_lines[dum1].split()
                            #print(elements)
                            for j in range(end):
                                s = k*5 + j
                                m = i + k*5
                                densCI_AO_a[m,s] = float(elements[j+1])
                                if (i != j):
                                    densCI_AO_a[s,m] = densCI_AO_a[m,s]
                        shift += irange
                    except (IndexError, ValueError):
                        break
                densCI_AO = densCI_AO_a
                print('denmat:\n',densCI_AO)
            dipXCI[st1,st1] = -2*np.trace(np.matmul(densCI_AO,dipX))
            dipYCI[st1,st1] = -2*np.trace(np.matmul(densCI_AO,dipY))
            dipZCI[st1,st1] = -2*np.trace(np.matmul(densCI_AO,dipZ))
            print('dipole moments of state {}: {}*x, {}*y, {}*z'.format(st1,dipXCI[st1,st1],dipYCI[st1,st1],dipZCI[st1,st1]))

with open(path+ident+'_CI_dimat.npz','wb') as f:
    np.save(f,dipZCI)


