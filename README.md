# Data generation


## Extracting CI coefficients from the logfiles
Begin by running `ci_coefficient_extractor_general.py` with the appropriate logfile for whichever molecular system ($HeH^{+}$ or $H_2$) in whichever basis (STO-3G or 6-31G). This will save the core Hamiltonian $H_0$ and the CI coefficients needed to assemble $\widetilde{B}$.

## Assembling $\widetilde{B}$
After running and saving the CI coefficients and core Hamiltonian from  `ci_coefficient_extractor_general.py`, open `products.py`. For whichever molecular system and basis you are in, change the appropriate ```logfile``` parameter at the top of the script. Subsequently, if you are in STO-3G, uncomment line 50. If you are in 6-31G, leave line 51 uncommented. These represent all possible states for the system. In lines 111 - 113, uncomment the appropriate CI coefficients for the molecular system that you are in (CI coefficients are saved to disk from `ci_coefficient_extractor_general.py`). Running the script will then generate and save $\widetilde{B}$.

## Generating the dipole moment matrix
`calc_tdm_casscf.ipynb` generates the correct dipole moment matrix $M$ from a given logfile. In the 4th cell, provide the path to whichever logfile is desired.

### For reference, all data generated via this procedure is provided in the `data` folder.


# Running the memory model


## No striding
If you are interested in running the memory model for any of our molecular systems using no striding, then the `jaxprop` folder provides JAX GPU compiled code for Equation (52) in the paper. Currently, each molecular system and basis are broken up into their own Jupyter notebooks. In the 2nd cell, load the appropriate $\widetilde{B}$ tensor for the parameter `P` and the dipole matrix $M$ into `dipmat`. In the following cell, load in the corresponding core Hamiltonian into `ham`. In the following cell, adjust the value for $\Delta t$ into the parameter `mydt`. The following cells then assemble the applied electric field, which can also be adjusted according to a user's preference (ex: change the frequency). The propagation of the time-dependent coefficients is then carried out. In the cell that defines the parameter `ells`, adjust and apply the range of values $\ell$ to be used in the propagation scheme. Following this, the symmetrized propagation scheme is formed. Finally, in the cell that begins with `for ell in ells:`, adjust the `numsteps` parameter to set a final time $T$. What will be computed for each value of $\ell$ is the MSE, residual error, condition number of the $M''(t)$ matrix at time $T$, and the propagated 1RDMs.

## Striding
If you are interested in running the memory model for any of our molecular systems using striding, then the `striding` folder provides CuPy GPU compiled code for Equation (52) in the paper. Similar to the JAX code, each molecular system and basis are in their own Jupyter notebook. Accordingly, the parameters mentioned above can be adjusted and loaded in for whichever molecular system is of interest. Towards the end of each notebook, `numsteps` can be adjusted to set a final time $T$. The `strides` parameter can be adjusted to include varying striding lengths $k$ to be applied for a fixed history length $\ell$. The fixed history length $\ell$ is chosen at the end of the line `ells` assignment. The propagation scheme will then be ran for all the different strides corresponding to the fixed value of $\ell$. Similarly, the MSE, condition number of the $M''(t)$ at time $T$, and propagated 1RDMs will be computed for the parameters chosen.
