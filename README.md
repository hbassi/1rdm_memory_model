# Data generation

First, carry out these preliminary steps.

## Extracting CI coefficients from the logfiles
To extract CI coefficients, run
```
python extract_ci_info.py --mol MOL --basis BASIS
```
where MOL can be either `heh+` or `h2` and BASIS can be either `sto-3g` or `6-31g`.  This will save the core Hamiltonian $H_0$ and the CI coefficients needed to assemble $\widetilde{B}$.

In the above code, and in the next two codes, there are two optional arguments, `--inpath` and `--outpath`; we need not touch these unless we want to load and store data in a location other than the `logfiles` subdirectory.

After running the above code, you should see two new files with filenames that end with `_hamiltonian.npy` and `_ci_coefficients.npy`, both in the `logfiles` subdirectory (unless you chose a different `outpath`).

## Assembling $\widetilde{B}$
To assemble this tensor, run
```
python compute_btensor.py --mol MOL --basis BASIS
```
where MOL can be either `heh+` or `h2` and BASIS can be either `sto-3g` or `6-31g`.  This will create a file that ends with `_tensor.npy` in the `logfiles` subdirectory (unless you chose a different `outpath`).

## Generating the dipole moment matrix
To generate the dipole moment matrix in the z direction, run
```
python compute_dipmat.py --mol MOL --basis BASIS
```
where MOL can be either `heh+` or `h2` and BASIS can be either `sto-3g` or `6-31g`.  This will create a file that ends with `_CI_dimat.npz` in the `logfiles` subdirectory (unless you chose a different `outpath`).

### For reference, all data generated via this procedure is provided in the `data` folder.


# Running the memory model


## No striding
If you are interested in running the memory model for any of our molecular systems using no striding, then the `jaxprop` folder provides JAX GPU compiled code for Equation (52) in the paper. Currently, each molecular system and basis are broken up into their own Jupyter notebooks. In the 2nd cell, load the appropriate $\widetilde{B}$ tensor for the parameter `P` and the dipole matrix $M$ into `dipmat`. In the following cell, load in the corresponding core Hamiltonian into `ham`. In the following cell, adjust the value for $\Delta t$ into the parameter `mydt`. The following cells then assemble the applied electric field, which can also be adjusted according to a user's preference (ex: change the frequency). The propagation of the time-dependent coefficients is then carried out. In the cell that defines the parameter `ells`, adjust and apply the range of values $\ell$ to be used in the propagation scheme. Following this, the symmetrized propagation scheme is formed. Finally, in the cell that begins with `for ell in ells:`, adjust the `numsteps` parameter to set a final time $T$. What will be computed for each value of $\ell$ is the MSE, residual error, condition number of the $M''(t)$ matrix at time $T$, and the propagated 1RDMs.

## Striding
If you are interested in running the memory model for any of our molecular systems using striding, then the `striding` folder provides CuPy GPU compiled code for Equation (52) in the paper. Similar to the JAX code, each molecular system and basis are in their own Jupyter notebook. Accordingly, the parameters mentioned above can be adjusted and loaded in for whichever molecular system is of interest. Towards the end of each notebook, `numsteps` can be adjusted to set a final time $T$. The `strides` parameter can be adjusted to include varying striding lengths $k$ to be applied for a fixed history length $\ell$. The fixed history length $\ell$ is chosen at the end of the line `ells` assignment. The propagation scheme will then be ran for all the different strides corresponding to the fixed value of $\ell$. Similarly, the MSE, condition number of the $M''(t)$ at time $T$, and propagated 1RDMs will be computed for the parameters chosen.
