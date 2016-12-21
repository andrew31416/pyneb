An interface to ase's nudged elastic band (NEB) hamiltonian and associated minimisers. pyneb.py is intended to 
generate each image of each NEB iteration successively, after CASTEP DFT calculations have been run on the 
previous band in the working directory. Intended for large systems where continuous NEB calculations are not 
possible or prefered on HPC clusters, allowing the user to submit individual image calculations to the cluster at 
a time.

------------
DEPENDENCIES
------------

1. Atomic Simulation Environment (ASE)

2. Docopt

-------
INSTALL
-------

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

6. Now mv atoms.py from this (pyneb) distribution, to the local ASE root

7. Now mv optimize.py from this (pyneb) distribution, to the local ASE root

---
RUN
---

1. Construct two seed structures, the end points of your NEB and put the .cell and .param files in an empty 
   working directory.

2. Initialise an initial image by running pyneb in the working directory with appropriate arguments. See:
   >>> pyneb.py -h

3. Run CASTEP on the first band, <calc-name>_1-i, for i=[1,<N>] for <N> images per band.

4. Run pyneb,
   >>> pyneb.py
   in the working directory to generate the next band, <calc-name>_2-i, for i=[1,<N>].

5. Repeat steps 3. and 4. ...


