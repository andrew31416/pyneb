"""
routines for dealing with parsing an ase Atoms object to a .cell file
"""


def fetchkeywords(_cellfile):
    """
    strip lattice vector and atom position coordinates
    """
    # for importing pyneb library destination
    import pyneb

    # keys not to copy 
    protected_keys = ['lattice_cart','lattice_abc','positions_frac','positions_abs']

    with open(pyneb.pyneb_lib+'pyneb_cellkeywords.txt','r') as f:
        keywords = f.readlines()

        # strip newline characters
        for i,_key in enumerate(keywords):
            keywords[i] = _key.lower().strip('\n')

    with open(_cellfile,'r') as f:
        flines = f.readlines()

    # list of lines of keyword extracts to copy
    keylines = []

    for _key in keywords:
        if _key in protected_keys:
            continue
        
        instances = []

        for i,line in enumerate(flines):
            if _key in line.lower():
                instances.append(i)

        if len(instances)==0:
            continue
        elif len(instances)==1:
            keylines.append(flines[instances[0]]+'\n')
        elif len(instances)==2:
            for i in range(instances[1]-instances[0]+1):
                keylines.append(flines[instances[0]+i])
            keylines.append('\n')
        elif len(instances)>2:
            raise ValueError('{} keyword found {} many times. Parsing error.'.\
            format(_key,len(instances)))
    
    return keylines

def comparekeywords(lines):
    """
    compre 2 lists of lines to be identical when space delimited
    """
    from copy import deepcopy
    import pyneb

    with open(pyneb.pyneb_lib+'pyneb_cellkeywords.txt','r') as f:
        keywords = f.readlines()

        # strip newline characters
        for i,_key in enumerate(keywords):
            keywords[i] = _key.lower().strip('\n')
    
    # list of dictionary of keywords
    dictlist = []

    for i in range(2):
        # list of lines belonging to file i
        flines = lines[i]

        # list of lines of keyword extracts to copy
        keylines = {}

        for _key in keywords:
            
            instances = []
            tmplines = []

            for i,line in enumerate(flines):
                if _key in line.lower():
                    instances.append(i)
        

            if len(instances)==0:
                continue
            elif len(instances)==1:
                tmplines.append(flines[instances[0]]+'\n')
            elif len(instances)==2:
                for i in range(instances[1]-instances[0]+1):
                    tmplines.append(flines[instances[0]+i])
                tmplines.append('\n')
            elif len(instances)>2:
                raise ValueError('{} keyword found {} many times. Parsing error.'.\
                format(_key,len(instances)))
        
            keylines.update({_key:tmplines})
        
        dictlist.append(deepcopy(keylines))

    # make sets out of present keywords, the compare
    sets = [set([]),set([])]
    for i in range(2):
        for _key in dictlist[i]:
            sets[i].update(set([_key]))
    
    if sets[0]!=sets[1]:
        raise ValueError('initial .cell seedfiles do not have identical keyword values')

    # now have list of dictionaries, compare keys
    for _key in dictlist[0]:
        for i in range(len(dictlist[0][_key])):
            if len(dictlist[0][_key][i].split()) != 0 and len(dictlist[1][_key][i].split()) !=0 :
                if dictlist[0][_key][i].lower().split() != dictlist[1][_key][i].lower().split():
                    raise ValueError('initial .cell seedfiles do not have identical keyword values: \n {}\n{}'.\
                    format(dictlist[0][_key][i],dictlist[1][_key][i]))

def celltoase(_file):
    """
    return an ase Atoms object for the .cell file, _file
    """
    from ase.atom import Atom
    from ase.atoms import Atoms
    from ase.constraints import FixCartesian
    from numpy import zeros

    def fractocart(gamma,cell):
        """
        convert fractional to cartesian coordinates
        
        assumes cell[i,j] is the jth cartesian component of the ith cell vector 
        """
        r = [  sum([gamma[j]*cell[j,i] for j in range(3)]) for i in range(3) ]
        
        return r
    ############

    def calc_index(elements,line):
        """
        return the atom index i=[0,N-1] of a constraint,line
        with atom number in given species line.split()[1], line.split()[2]

        elements is the list of atom species with which to order
        """
        # species element
        species = line.split()[1].lower()

        # atom number with species
        species_num = int(line.split()[2]) - 1

        # list of atom indices for each instance of species atom type
        species_idxs = []

        for i,_atm_type in enumerate(elements):
            if _atm_type.lower() == species:
                species_idxs.append(i)
        
        return species_idxs[species_num]

    # check format
    assert _file.split('.')[-1].lower()=='cell','internal error: {} is not a .cell file!'.format(_file)
    
    with open(_file,'r') as f:
        flines = f.readlines()

        # flines indices for lattice cart, atoms and cartesian constraints
        lat_lines = []
        atm_lines = []
        car_lines = []

        for i,l in enumerate(flines):
            if 'lattice_cart' in l.lower():
                lat_lines.append(i)
            elif 'positions_frac' in l.lower():
                atm_lines.append(i)
            elif 'ionic_constraints' in l.lower():
                car_lines.append(i)

        assert len(lat_lines)==2 and len(atm_lines)==2 and (len(car_lines)==2 or len(car_lines)==0),\
        'parsing error for {}, please check file.'.format(_file)

        # get cell vectors cell[i,j] is the jth cartesian component of the ith cell vector
        cell = zeros((3,3),dtype=float)

        for i,line in enumerate(flines[lat_lines[0]+1:lat_lines[1]]):
            for j,coordinate in enumerate(line.split()):
                cell[i,j] = float(coordinate)

        # fetch number of atoms in .cell file
        Natm = 0
        for line in flines[atm_lines[0]+1:atm_lines[1]]:
            if len(line.split())>0:
                Natm += 1
        
        elements = []

        # get atom species list,elements
        for line in flines[atm_lines[0]+1:atm_lines[1]]:
            if len(line.split())!=0:
                elements.append(line.split()[0])


        # no constraint if unspecified, signified by zeros
        constraints = [zeros(3,dtype=int) for i in range(Natm)]

        if len(car_lines)!=0:
            for line in flines[car_lines[0]+1:car_lines[1]]:
                if len(line.split())>0:
                    assert len(line.split())==6,'error parsing constraint line: {}'.format(line)

                    # list index of constraint
                    index = calc_index(elements,line)
                    #index = int(line.split()[2])-1 deprecated

                    for i in range(3):
                        val = float(line.split()[3+i])
                        if val!=0:
                            assert val==1,'unsupported constraint value. Only 1 or 0 supported.' 

                            constraints[index][i] = 1


        # list for ase atom objects
        atoms_list = []

        # list of ase constraints objects
        constraint_list = []

        # allow for spurious newlines between atom entries
        cntr = 0

        for line in flines[atm_lines[0]+1:atm_lines[1]]:
            if len(line.split())>0:
                # cartesian components of atom
                _xyztuple = ( fractocart( [float(_frac) for _frac in line.split()[1:]],cell ) )
                
                # atomic species of atom
                species = line.split()[0]

                # cartesian constraints on atom, if any
                if any(constraints[cntr]):
                    # cartesian constraints, if mask[i]==1, ith cartesian component should be fixed
                    constraint_list.append(FixCartesian(cntr,mask=tuple(constraints[cntr])))

                atoms_list.append(Atom(symbol=species,position=_xyztuple))
                
                # book keeping
                cntr += 1

        return Atoms(atoms_list,cell=cell,pbc=(True,True,True),constraint=constraint_list)


def asetocell(atoms_object,keywords,cellfile):
    """
    extralines is a list of lines to be appended to the lattice
    vector and atom positions provided by atoms_object
    """
    def _check_position(fracpos,cell,tol=0.2):
        """
        rescale fractional coordinates to be between 0 and 1 and check
        that no 2 atoms are closer than tol angstrom apart
        """
        import numpy as np

        for i,_pos in enumerate(fracpos):
            if any([_v >1.0 or _v < 0.0 for _v in _pos]):
                # a vector coordinate needs mapping back
                for j in range(3):
                    fracpos[i,j] = fracpos[i,j] - round(fracpos[i,j])


        # check that no 2 atoms overlap
        for i,_posi in enumerate(fracpos):
            for j,_posj in enumerate(fracpos):
                if j > i:
                    assert isinstance(_posi,np.ndarray) and isinstance(_posj,np.ndarray),\
                    'ase Atoms object fractional coordinates are not np.ndarrays'

                    # check displacement
                    disp = _posi - _posj

                    # round to nearest image
                    disp -= np.round(disp)

                    # convert to cartesians
                    cart = [0.0,0.0,0.0]

                    for k in range(3):
                        cart[k] = sum([disp[m]*cell[m,k] for m in range(3)])
                    
                    assert np.sqrt(sum([_c**2 for _c in cart]))>tol,'atoms {},{} are overlapping with a cartesian dispalcement of : {}'.\
                    format(i+1,j+1,cart)

        return fracpos

    # append cell vectors
    flines = ['%BLOCK LATTICE_CART\n']
    for _c in atoms_object.cell:
        flines.append('{:12.6f} {:12.6f} {:12.6f}\n'.format(_c[0],_c[1],_c[2]))
    flines.append('%ENDBLOCK LATTICE_CART\n\n')

    # append atom species and fractional coordinates
    frac_coords = atoms_object.get_scaled_positions(wrap=True)

    # check fractional coordinates
    frac_coords = _check_position(frac_coords,atoms_object.cell,tol=0.2)

    elements = []
    for i in atoms_object:
        elements.append(i.symbol)

    flines.append('%BLOCK POSITIONS_FRAC\n')
    for i,_pos in enumerate(frac_coords):
        flines.append(elements[i]+'  {:12.6f} {:12.6f} {:12.6f}\n'.format(_pos[0],_pos[1],_pos[2]))
    flines.append('%ENDBLOCK POSITIONS_FRAC\n\n')

    flines += keywords

    # write cell file to cellfile
    with open(cellfile,'w') as f:
        f.writelines(flines)

def checkparamfiles(param1,param2):
    """
    compare 2 .param files allowing for deviations in space delimiting
   
    check for singlepoint and atomic force calculations
    """
    with open(param1,'r') as f:
        flines1 = f.readlines()
    with open(param2,'r') as f:
        flines2 = f.readlines()

    flines = [ flines1,flines2 ]

    # list for dictionaries of .param keyword:value pairs
    dicts = [{},{}]

    for i in range(2):
        for _l in flines[i]:
            if len(_l)!=0:
                # create dictionary of .param {keywords:value} for both files, store these dicts in a list
                dicts[i].update({_l.split(':')[0].split()[0].lower() : _l.split(':')[1].split()[0].lower()})

    # create sets of keywords
    setlist = [set([]),set([])]

    for i in range(2):
        for _key in dicts[i]:
            setlist[i].update(set([_key]))

    # check keyword:values pairs are the same
    assert setlist[0]==setlist[1],'.param files {}, {} are not identical'.format(param1,param2)
    assert all([dicts[1][_key]==dicts[0][_key] for _key in dicts[0]]),\
    '.param files {}, {} are not identical'.format(param1,param2)

    # check that calculation type is singlepoint
    assert 'task' in dicts[0],'calculation type not specified in {},{}'.format(param1,param2)
    assert dicts[0]['task'] == 'singlepoint','calculation task specified must be a singlepoint'

    # check that forces are being computed
    assert 'calculate_stress' in dicts[0],\
    '"calculate_stress : true" must be included in {}, {} to output forces'.format(param1,param2)
    assert dicts[0]['calculate_stress']=='true',\
    '"calculate_stress : true" must be included in {}, {} to output forces'.format(param1,param2)


def checkimages(atoms1,atoms2):
    """
    check atoms objects to see that they are not identical!
    """
    import numpy as np

    different = False

    for i,_atom1 in enumerate(atoms1):
        if np.array_equal(_atom1.position,atoms2[i].position) is not True:
            different = True
    
    assert different,'initial configurations appear to be identical!'


def adoptcellorder(castep_atoms,cell_atoms):
    """
    switch the order of atoms in a castep Atoms object to be the same order
    as that in a cell Atoms object
    """
    from numpy import zeros as zeros
    from numpy import isclose as isclose
    from ase.atom import Atom
    from ase.atoms import Atoms

    # tolerence
    tol = 0.0001

    # sanity check
    check = [0]*len(castep_atoms.positions)

    atoms_list = [None for i in range(len(castep_atoms.positions))]

    forces = zeros((len(cell_atoms.positions),3),dtype=float,order='C')

    for i,_cellpos in enumerate(cell_atoms.get_scaled_positions(wrap=True)):
        for j,_cstppos in enumerate(castep_atoms.get_scaled_positions(wrap=True)):
            # need to account for cases when atom can have fractional coordinate of 0 or 1           
            if all([any([isclose(_cstppos[k]-1,_cellpos[k],atol=tol),
                         isclose(_cstppos[k]+0,_cellpos[k],atol=tol),
                         isclose(_cstppos[k]+1,_cellpos[k],atol=tol)]) for k in range(3)]):
            #if all([isclose(_cstppos[k],_cellpos[k],atol=0.0001) for k in range(3)]):
                # copy atom forces
                forces[i][:] = castep_atoms.forces[j][:]
                
                # copy atom object
                atoms_list[i] = cell_atoms[i]

                # sanity check
                check[i] += 1

    assert all([_c ==1 for _c in check]),'error reordering .castep atoms'

    # create new Atoms objects, borrow constraints from .cell file
    return Atoms(atoms_list,cell=cell_atoms.cell,pbc=True,castep_neb=True,\
    system_energy=castep_atoms.energy,system_forces=forces,constraint=cell_atoms.constraints)

