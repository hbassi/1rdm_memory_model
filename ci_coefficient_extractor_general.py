import re
import numpy as np
import argparse
import sys

parser = argparse.ArgumentParser(prog='ci_coefficient_extractor',
                                 description='extract CI coefficients and Hamiltonian from log file')

parser.add_argument('--mol', required=True, help='molecule')
parser.add_argument('--basis', required=True, help='basis')
parser.add_argument('--path', required=False, help='custom path to log files')

# actually parse command-line arguments
args = parser.parse_args()

mol = args.mol
basis = args.basis

# set path to log files
if args.path:
    path = args.path
else:
    path = './logfiles/'

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
f = open(path+ident+'.log','r')
extract = False
eigenvalues = False
keepcounting = True
rownumber = 0
rows = []
for x in f.read().strip().split('\n'):
    if 'kranka test CI' in x:
        extract = True
        continue
    if extract:
        if 'EIGENVALUES' in x:
            break
        if x == '\n' or '':
            continue
        rows.append(x)
        

input_string="\n".join(rows)
chunks = input_string.strip().split('\n\n')
extracted_coefficients = []
pattern = r"[-+]?\d*\.\d+(?:[Ee][-+]?\d+)?"
for chunk in chunks:
    lines = chunk.strip().split('\n')
    coefficients_chunk = []
    for line in lines:
        numerical_values = re.findall(pattern, line)
        if numerical_values:
            coefficients_chunk.extend([float(value) for value in numerical_values])

    extracted_coefficients.append(coefficients_chunk)

final_coefficients = []
hamiltonian = []
for i, coefficients_chunk in enumerate(extracted_coefficients, start=1):
    print(f"Chunk {i} coefficients:")
    for coefficient in coefficients_chunk:
        print(coefficient, end=" ")
    print("\n")
    hamiltonian.append(coefficients_chunk[0])
    final_coefficients.append(coefficients_chunk[1:])
final_coefficients = np.array(final_coefficients)
hamiltonian = np.array(hamiltonian)
hamiltonian = np.diag(hamiltonian)
print(final_coefficients.shape)
print(hamiltonian.shape)

print(final_coefficients)
print(hamiltonian)
coeff_fname = ident + '_ci_coefficients.npy'
np.save(coeff_fname, final_coefficients)
ham_fname = ident + '_hamiltonian.npy'
np.save(ham_fname, hamiltonian)

