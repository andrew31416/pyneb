A wrapper for ASE's nudged elastict band (NEB) methods, providing I/O parsing for CASTEP. Intended for large configurations, pyneb allows the user to run (independant) single point calculations on individual images with CASTEP. Once all single point calculations for a given band are complete, running pyneb in your working directory will generate .cell and .param files for the next band.

Initial seed (.cell and .param) files are assumed to be the two metastable (end) points of the band - this is not checked for and if the configurations have not been geometry optimised for the seed calculation parameters, spurious minimum energy pathways may result!

------------
DEPENDENCIES
------------

1. Atomic Simulation Environment (ASE)
> conda install -c jochym ase=3.11.0 

2. Docopt
> conda install -c asmeurer docopt=0.6.2 

-------
INSTALL
-------

To make use of ASE's NEB methods, some modifications are necessary to atoms.py and optimize.py. Please read the INSTALL guide carefully!


1. Move the library files;
    
    - pyneb_cellkeywords.txt
    - pyneb_parsecell.py
    - pyneb_parser_castep.py
    - pyneb_parser_general.py
    - pyneb_structureformat.py
   
   to any desired space on the hard disk, <lib_dir>

2. Open pyneb.py and append <lib_dir> to sys.path

3. cd into the root of your local ASE distribution

4. In the local ASE root, mv atoms.py to atoms-original.py for safe keeping

5. In the local ASE root, mv optimize.py to optimize-original.py for safe keeping

6. Now soft link or move atoms.py from this (pyneb) distribution, to the local ASE root

7. Now soft link or move optimize.py from this (pyneb) distribution, to the local ASE root

---
RUN
---

1. Construct two seed structures, the end points of your NEB and put the .cell and .param files in an empty 
   working directory. Ensure these are geometry optimised to a reasonable tolerance.

2. Initialise the first band by running pyneb in the working directory with appropriate arguments. See:
   > pyneb.py -h
   
   for details.
   
3. Run CASTEP on the jth band, calc-name_j-i, for i=[1,<N>] for <N> images per band.

4. Once all single points calculations for the most recent band (j) are complete, run
   > pyneb.py
   
   in the working directory to generate the next band, calc-name_{j+1}-i, for i=[1,<N>].

5. Repeat steps 3. and 4. until you are satisfied that the NEB hamiltonian has been minimised.

-------
EXAMPLE
-------

Minimum energy pathway between pristine and 5-7 defective graphene

![place holder 1](andrew031191.github.com/pyneb/examples/graphene_5-7/placeholder.png)


