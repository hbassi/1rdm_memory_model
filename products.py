import re
import numpy as np

#f = open('./logfiles/casscf24_s15_heh+_6-31g.log','r')
#f = open('./logfiles/casscf22_s2_heh+_sto-3g.log','r')
#f = open('./logfiles/casscf22_s2_h2_sto-3g.log','r')
f = open('./logfiles/casscf24_s15_h2_6-31g.log','r')
def products():
    extract = False
    eigenvalues = False
    keepcounting = True
    rownumber = 0
    rows = []
    for x in f:
        if "Molecular Orbital Coefficient" in x:
            extract = True
        if extract:
            if "Eigenvalues" in x:
                eigenvalues = True
                # want to move to next line
                continue
            if eigenvalues:
                strlist = re.findall("-?\d+\.\d+", x)
                # stop parsing if there are no floats in the line
                if strlist == []:
                    eigenvalues = False
                    keepcounting = False
                else:
                    rows.append( np.loadtxt( strlist , ndmin=1) )
                    if keepcounting:
                        rownumber += 1

        if "Alpha Density Matrix" in x:
            extract = False

    #print(rows)
    #print(rownumber)

    # process the list of row vectors that we extracted
    chunks = []
    numchunks = len(rows)//rownumber
    for c in range(numchunks):
        si = c*rownumber
        ei = (c+1)*rownumber
        chunks.append( np.array(rows[si:ei]) )

    MOs = np.hstack(chunks).T

    #read states in from file under 'SLATER DETERMINANT BASIS'
    #states = ['10','ba','ab','01']
    states = ['1000','ba00','ab00','b0a0','0100','a0b0','b00a','0ba0','0ab0','a00b','0b0a','0010','0a0b','00ba','00ab','0001']


    def process(state):
        # check if we have a doubly occupied orbital
        if '1' in state:
            loc = state.index('1')
            return loc+1, loc+1
        else: # we have singly occupied orbitals
            loc1 = state.index('a')
            loc2 = state.index('b')
            return loc1+1, loc2+1
    products = {}
    # compute all reduced 1-electron integrals of all products of states
    numstates = len(states)
    for i in range(numstates):
        for j in range(numstates):
            i1, j1 = process(states[i])
            i2, j2 = process(states[j])
            # "do" the integral
            outer_product = states[i] + ' * ' + states[j]
            symb_op_1 = ''
            symb_op_2 = ''
            if j1==j2:
                if (j1 < i1 and j2 < i2) or (j1 >= i1 and j2 >= i2):
                    symb_op_1 = "|" + str(i1) + " >< " + str(i2) + "|"
                else:
                    symb_op_1 = "-|" + str(i1) + " >< " + str(i2) + "|"
            if i1==i2:
                if (j2 < i2 and j1 < i1) or (j2 >= i2 and j1 >= i1):
                    symb_op_2 = "|" + str(j1) + " >< " + str(j2) + "|"
                else:
                    symb_op_2 = "-|" + str(j1) + " >< " + str(j2) + "|"
            products[outer_product] = symb_op_1 + symb_op_2
            #print("\n")
    #print(products)  
    final_products = {}

    for key in list(products.keys()):
        prod = products[key]
        if prod == '':
            final_products[key] = np.zeros((MOs.shape[0],MOs.shape[0]))
        else:
            num_minuses = prod.count("-")
            if num_minuses == 1:
                scaling = -1.0
                print(key)
            else:
                scaling = 1.0
            #scaling = 1.0
            intlist = re.findall(r"-?\d+", prod)
            int_values = [int(value) for value in intlist]
            if len(int_values) == 2*MOs.shape[0]:
                final_products[key] = scaling*(0.5*(MOs[int_values[0]-1].reshape((MOs.shape[0],1)) @ MOs[int_values[1]-1].reshape((MOs.shape[0],1)).T) + 0.5*(MOs[int_values[2]-1].reshape((MOs.shape[0],1)) @ MOs[int_values[3]-1].reshape((MOs.shape[0],1)).T))
            else:
                final_products[key] = scaling * 0.5*(MOs[int_values[0]-1].reshape((MOs.shape[0],1)) @ MOs[int_values[1]-1].reshape((MOs.shape[0],1)).T)
    #print(final_products)



    #ci_coeffs = np.load('casscf24_s15_h2_6-31g_ci_coefficients.npz').T
    #ci_coeffs = np.load('casscf24_s15_heh+_6-31g_ci_coefficients.npz').T
    ci_coeffs = np.load('casscf22_s2_heh+_sto-3g_ci_coefficients.npz').T
    mapping_ci_coeffs = {}
    import pdb;pdb.set_trace()
    ctr = 0
    for key in states:
        mapping_ci_coeffs[key] = ci_coeffs[ctr]
        ctr += 1


    tens = np.zeros([ci_coeffs.shape[0],ci_coeffs.shape[1],MOs.shape[0],MOs.shape[1]])

    for i in range(0,len(states)):
        for j in range(0,len(states)):
            for key_mat in list(final_products.keys()):
                mat = final_products[key_mat]
                pattern = r'(.*) \* (.*)' 
                match = re.match(pattern, key_mat)
                state_i = match.group(1)
                state_j = match.group(2)
                c_i = mapping_ci_coeffs[state_i][i]
                c_j = mapping_ci_coeffs[state_j][j]
                tens[i,j,:,:] += (c_i * c_j)* mat
            tens[i,j,:,:] *= 2.0  
        
    return tens

if __name__ == '__main__':
    tens = products()
    # with open('casscf24_s15_h2_6-31g_tensor.npz', 'wb') as f:
    #     np.save(f,tens)
    