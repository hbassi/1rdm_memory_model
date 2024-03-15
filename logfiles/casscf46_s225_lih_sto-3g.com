%chk=casscf46_lih_sto-3g.chk
!%oldchk=sa-casscf25_s0_s0=1.chk
%subst l510 /home/kranka/proj_isborn/gdv_links_KR/gdvi14+/l510 
#P CASSCF(4,6,fulldiag,NRoot=225,SaveGEDensities)/sto-3g scf(tight,maxcyc=100) nosymm scf(tight,maxcyc=1500,conventional) iop(3/33=6) extralinks(l316,l308) noraff symm=noint iop(3/33=3) pop(full) iop(6/8=1,3/33=3,3/36=1,4/33=3,5/33=3,6/33=3,9/33=3,5/72=-3)

CAS(4,6)SCF/STO-3G, highest singlet state single point, pert. calc. (all density matrices printed)

0 1
 H       0.000000    0.000000   -0.765000
 Li      0.000000    0.000000    0.765000

