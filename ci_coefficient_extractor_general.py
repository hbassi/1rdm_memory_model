import re
import numpy as np
#f = open('./logfiles/casscf24_s15_heh+_6-31g.log','r')
#f = open('./logfiles/casscf22_s2_heh+_sto-3g.log','r')
f = open('./logfiles/casscf24_s15_h2_6-31g.log','r')
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
with open('casscf24_s15_h2_6-31g_ci_coefficients.npz', 'wb') as f:
    np.save(f, final_coefficients)
f.close()
with open('casscf24_s15_h2_6-31g_hamiltonian.npz', 'wb') as f:
    np.save(f, hamiltonian)
f.close()