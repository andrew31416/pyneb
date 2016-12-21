#!/bin/env python3

import numpy as np
import os, sys, warnings
import ase
import copy
from time import time
from pyneb_structureformat import supercell
import warnings

def deprecate(fun):
    warnings.warn("This function ({} > {}) will be removed in future updates!".format(fun.__module__,fun.__name__),DeprecationWarning)
    return fun

# CODATA2002: default in CASTEP 5.01
# (-> check in more recent CASTEP in case of numerical discrepancies?!)
# taken from
#    http://physics.nist.gov/cuu/Document/all_2002.pdf
units_CODATA2002 = {
    'hbar': 6.58211915E-16,     # eVs
    'Eh': 27.2113845,           # eV
    'kB': 8.617343E-5,          # eV/K
    'a0': 0.5291772108,         # A
    'c': 299792458,             # m/s
    'e': 1.60217653E-19,        # C
    'me': 5.4857990945E-4}      # u    

units_CODATA2002['t0'] = units_CODATA2002['hbar']/units_CODATA2002['Eh']
units_CODATA2002['Pascal'] = units_CODATA2002['e']*1E30
    
def convert_gridtocartesian(gridnum,cell,invNpoints):
    """
    convert grid numbers gridnum[0:3] to cartesian positions
    """
    cart = np.zeros(3,dtype=float)
    
    for ia in range(3):
        for ib in range(3):
            cart[ia] += (gridnum[ib]-1.0)*invNpoints[ib]*cell[ib][ia]
    return cart
    
def castep_safety_checks(lines):
    for line in lines:
        # check that all SCF cycles converged
        assert ('Warning: electronic minimisation did not converge' not in line),\
        ('SCF cycle has not converged for '+fd.name)
        # check geometry optimization has converged
        assert ('WARNING - Geometry optimization failed to converge' not in line),\
        ('geometry optimization has not converged for '+fd.name)
        # check geometry optimisation has been performed if stated
        if clc_type == 'geometry optimization':
            assert ('WARNING - there is nothing to optimise - skipping relaxation' not in line),\
            ('geometry optimization has not been performed for '+ fd.name)

    # check units are as expected
    supp_units = {'length unit':[' A\n'],'energy unit':[' eV\n'],'force unit':[' eV/A\n'],\
    'pressure unit':[' GPa\n']}
    E2 = 'calculation unit is not supported. Supported units are eV,A,GPa.'
    for line in lines:
        if ('output' in line and len(line.split())>3):
            if line.split()[1]=='length' and line.split()[2]=='unit' :
                assert (line.split(':')[-1] in supp_units['length unit']),E2
            elif 'energy unit' in line:
                assert (line.split(':')[-1] in supp_units['energy unit']),E2
            elif 'force unit' in line:
                assert (line.split(':')[-1] in supp_units['force unit']),E2
            elif 'pressure unit' in line:
                assert (line.split(':')[-1] in supp_units['pressure unit']),E2
                
def _castep_castep_get_indices(clc_type,lines):
    
    idx = [iv for iv,v in enumerate(lines) if v ==  " +-------------------------------------------------+"]
    if len(idx)>3:
        warnings.warn("Found more than one calculation, reading only the last one...")
    #print("idx {}".format(idx))
    readstart = idx[-3]
    readend = len(lines)
    return readstart, readend, len(idx)>3

def castep_completemeness_checks(fd,sedc_energy_total,sedc_free_energy_total,
                                 sedc_energy_0K,free_energy_total,energy_total,
                                 energy_0K,supercells,atoms,species,forces,clc_type):
    E1 = 'unexpected behaviour in '+fd.name+' , check file for corruption'
    
    num_ens = [len(energy_total),len(energy_0K),len(free_energy_total)]
    num_ens_dc = [len(sedc_energy_total),len(sedc_free_energy_total),len(sedc_energy_0K)]
    num_config = [len(forces),len(supercells),len(atoms),len(species)]
    if not len(set(num_ens))==1:
        raise AssertionError("Mismatching numbers of energies:\n"+\
                             "Final energy, E ({})\n".format(len(energy_total))+\
                             "NB est. 0K energy (E-0.5TS) ({})\n".format(len(energy_0K))+\
                             "Final free energy (E-TS) ({})".format(len(free_energy_total)))
    if num_ens_dc[-1]>0 and not len(set(num_ens_dc))==1:
        raise AssertionError("Mismatching numbers of dispersion corrected energies:\n"+\
                             "Dispersion corrected final energy ({})\n".format(len(sedc_energy_total))+\
                             "Dispersion corrected final free energy ({})\n".format(len(sedc_free_energy_total))+\
                             "NB dispersion corrected est. 0K energy ({})\n".format(len(sedc_energy_0K)))
    if clc_type != "geometry optimization":
        if not len(set(num_config))==1:
            raise AssertionError("Mismatching configuration:\n"+\
                                 "Forces ({}) Supercells ({}) Atoms ({}) Species ({})".format(*num_config))
    
def castep_get_unit_cells(lines,idx_unit_cell):
    num = len(idx_unit_cell)
    unitcells = [[]]*num
    for i,ix in enumerate(idx_unit_cell):
        unitcells[i] = [[float(_c) for _c in _line.split()[0:3]] for _line in lines[ix+3:ix+6]]
    return unitcells
    
def castep_get_forces(lines,idx_forces):
    num = len(idx_forces)
    forces = [[]]*num
    for j,ix in enumerate(idx_forces):
        for i,line in enumerate(lines[ix:]):
            if line == " ******************************************************" or\
                line == " ******************************************************************************":
                tmp_end = i-1
                break
        tmp_forces = [list(filter(None,v.split(' '))) for v in lines[ix+6:ix+tmp_end]]
        forces[j] = np.array([v[3:6] for v in tmp_forces],dtype=float)
    return forces

def castep_get_cell_content(lines,idx_cell_content):
    num = len(idx_cell_content)
    atoms, species = [[]]*num, [[]]*num
    for i,ix in enumerate(idx_cell_content):
        idx_tmp = []
        c = 0
        for j,line in enumerate(lines[ix:]):
            if "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" in line:
                idx_tmp.append(j)
                c +=1
            if c == 2: break
            
        config_lines = lines[ix+idx_tmp[0]+4:ix+idx_tmp[1]]
        config_lines = [list(filter(None,v.split(' '))) for v in config_lines]
        species[i] = [v[1] for v in config_lines]
        atoms[i] = np.array([v[3:6] for v in config_lines],dtype=float)
    return atoms,species
    
def castep_get_stress(lines,idx_stress):
    num = len(idx_stress)
    stress = [[]]*num
    for i,ix in enumerate(idx_stress):
        stress[i] = np.array([list(filter(None,v.split(' ')))[2:5] for v in lines[ix+6:ix+9]])
    return stress

def read_castep_castep(fd, index=None, safety_checks=False,check_completeness=True):
    """parse a .castep file of supported calculation type.

    Parameters
    ----------
    lines: list of str
    
    Returns
    -------
    - the list of supercell structures appropriate to the calculation type.
        - single point energy   : a single structure
        - geometry optimization : the final optimised structure only
        - molecular dynamics    : Nsteps many structures
        
    Notes
    -----
    Supported calculations types:
        - single point energy
        - geometry optimisation
        - molecular dynamics
        
    """
    ####
    # Under which circumstances appear dispersion corrected energies?
    ####
    
    # supported calculation types for reading
    supported_MD_envs = set(["nvt","nvp","nve"])
    supp_clcs = set(['single point energy','geometry optimization','molecular dynamics'])
    lines = list(map(lambda x: x.rstrip('\n'),fd.readlines()))
    # determine calculation type
    for i,line in enumerate(lines):
        if "type of calculation" in line:
            clc_type = ' '.join(list(filter(None,line.split(' ')))[4:])
            assert clc_type in supp_clcs, "Assertion failed - found unexpected '{}' calculation type. Expected one of {}".format(clc_type,supp_clcs)
            break
    if i == len(lines):
        raise ValueError("Tough luck... corrupted file ({}) found no 'type of calculation' line!".format(fd.name))
    #print("calculation type {}".format(clc_type))
    
    # check how many calculations are present in case of single point calculation
    run_idx = [iv for iv,v in enumerate(lines) if "Run started" in v]
    cntr = len(run_idx)
    
    if cntr > 1:
        warnings.warn('Warning : more than 1 calculation is present in {}, are considering only the last calculation'.format(fd.name))
    if cntr == 0: raise ValueError("Missing calculation or corrupted file in {}...".format(fd.name))
            
    if safety_checks:
        castep_safety_checks(lines)    

    # get indices for processing    
    readstart, readend, multiple_calc = _castep_castep_get_indices(clc_type,lines)
    #print("readstart {} readend {}".format(readstart,readend))
        
    # create a list of material properties for each configuration
    energy_total            = []    # 1. total energy
    free_energy_total       = []    # 2. total free energy
    energy_0K               = []    # 3. 0K total energy estimate
    sedc_energy_total       = []    # 4. sedc corrected total energy
    sedc_free_energy_total  = []    # 5. sedc corrected free energy
    sedc_energy_0K          = []    # 6. sedc corrected 0K energy estimate
    MP_energy_total         = []    # 7. Makov-Payne finite basis set corrected total energy
    supercells              = []    # 8. supercell vectors in cartesian
    atoms                   = []    # 9. fractional coordinates of atoms
    species                 = []    # 10. atom types
    forces                  = []    # 11. atom cartesian force components /(eV/A)
    stress                  = []    # 12. stress tensor / (GPa)
    spacegroup = None               # 13. space group : take number and Hermann-Mauguin notation
    charge = None                   # 14. atomic charge / (|e-|)
    enthalpy                = []    # 15. enthalpy H = U + PV

    #for ia in range(readstart,readend):
    ia = int(readstart)
    idx_unit_cell = []
    idx_forces = []
    idx_cell_content = []
    idx_stress = []
    idx_charge = []
    num_cutoff = 0 #for some reason castep may do a small cutoff convergence test leading to additional superfluous energy values
    while ia < readend:
            
        line = lines[ia]
        if 'Final energy, E' in line:
            energy_total.append(float(line.split()[-2]))
        elif 'Final free energy (E-TS)' in line:
            free_energy_total.append(float(line.split()[-2]))
        elif 'NB est. 0K energy (E-0.5TS)' in line:
            energy_0K.append(float(line.split()[-2]))
        elif 'Dispersion corrected final energy' in line:
            sedc_energy_total.append(float(line.split()[-2]))
        elif 'Dispersion corrected final free energy' in line:
            sedc_free_energy_total.append(float(line.split()[-2]))
        elif 'NB dispersion corrected est. 0K energy' in line:
            sedc_energy_0K.append(float(line.split()[-2]))
        elif 'Total energy corrected for finite basis set' in line:
            MP_energy_total.append(float(line.split()[-2]))
        elif 'Unit Cell' in line: #bloody hell, castep outputs multiple unit cells for NPT but only single for NVT...
            idx_unit_cell.append(ia)
        elif "********** Forces ********" in line \
                or line == " ***************** Symmetrised Forces *****************":
        #elif line == " *********************************** Forces ***********************************" \
            idx_forces.append(ia)
        elif "Cell Contents" in line:
            idx_cell_content.append(ia)
        elif line == " ************************ Molecular Dynamics Parameters ************************":
            for ib,line in enumerate(lines[ia:]):
                if "ensemble" in line:
                    break 
            md_env = list(filter(None,lines[ib+ia].split(" ")))[-1].lower()
        elif line == " *********** Symmetrised Stress Tensor ***********" or \
                line == " ***************** Stress Tensor *****************":
            idx_stress.append(ia)
        elif 'Final Enthalpy' in line:
            enthalpy.append(float(line.split()[4]))
        elif 'Space group of crystal' in line:
            spacegroup = {'number':int((line.split()[5]).split(':')[0]),\
                          'Hermann-Mauguin':(line.split()[6]).split(',')[0]}
        elif "Calculating finite basis set correction with" in line:
            num_cutoff = int(list(filter(None,line.split(' ')))[6])
            
        ia += 1
        
    #print("energy_total {} num_cutoff {}".format(len(energy_total),num_cutoff))
    if num_cutoff>0:
        energy_total = energy_total[num_cutoff-1:] 
        free_energy_total = free_energy_total[num_cutoff-1:] 
        energy_0K = energy_0K[num_cutoff-1:] 
    
    supercells = castep_get_unit_cells(lines,idx_unit_cell)
    atoms, species = castep_get_cell_content(lines,idx_cell_content)    
    forces = castep_get_forces(lines,idx_forces)
    stress = castep_get_stress(lines,idx_stress)
    
    if clc_type == "molecular dynamics":
        # in case of MD the last forces are repeated -> scratch those
        forces = forces[:-1]
        # also if only one stress tensor appears for MD then that's also only for the last configuration...
        if len(stress) == 1:
            stress = []
    # in case of NVT MD castep prints only one simulation box, hence need to multiply it...
    const_box_cases = set(["nvt"])  
    if clc_type == "molecular dynamics" and md_env in const_box_cases:
        for i in range(len(atoms)-1):
            supercells.append(supercells[0])
    
    if check_completeness:       
        castep_completemeness_checks(fd,sedc_energy_total,sedc_free_energy_total,
                                    sedc_energy_0K,free_energy_total,energy_total,
                                    energy_0K,supercells,atoms,species,forces,clc_type)

    # assign all attributes to in-house data structure
    structures = []

    if clc_type == "single point energy":
        tmp = supercell()
        tmp["cell"] = supercells[-1]
        tmp["positions"] = atoms[-1]        
        tmp["species"] = species[-1]

        if len(forces) > 0:
            tmp["forces"] = forces[-1]                
        if len(stress)>0:
            tmp["stress"] = stress[-1]          
        if len(sedc_energy_0K)>0:
            tmp["energy"] = sedc_energy_0K[-1]
        else:
            tmp["energy"] = energy_0K[-1]
        if spacegroup is not None: 
            tmp["spacegroup"] = spacegroup
        structname = fd.name.split('/')[-1]
        tmp["name"] = structname
        if len(enthalpy)>0:
            tmp["enthalpy"] = enthalpy[-1]    
        # add contributing file name
        tmp["files"] = [fd.name.split('/')[-1]]
        structures += [copy.deepcopy(tmp)]
    else:    
        for ia in range(len(supercells)):
            tmp = supercell()
            tmp["cell"] = supercells[ia]
            
            # fractional atomic positions
            if clc_type != 'geometry optimization':
                # final positions are not given for geom. opt.
                tmp["positions"] = atoms[ia]        
                tmp["species"] = species[ia]

            if len(forces) > 0:
                tmp["forces"] = forces[ia]            
            
            #TO CHECK! is the handling of the stress really ok this way?!
            if len(stress)==1 and ia == len(energy_total)-1:
                tmp["stress"] = stress[0]
                
            elif len(stress)>1:
                tmp["stress"] = stress[ia]          
            
            if len(sedc_energy_0K)>0:
                tmp["energy"] = sedc_energy_0K[ia]
            else:
                tmp["energy"] = energy_0K[ia]
            
            if spacegroup is not None: 
                tmp["spacegroup"] = spacegroup
            
            if len(energy_total)>1:
                fname = fd.name.split('/')[-1].split('.')
                fname = '.'.join(fname[:-1])+"-{}".format(ia)+"."+fname[-1]
                #print("energy {}: {}".format(fname,energy_total))
                structname = fname
            else:
                structname = fd.name.split('/')[-1]
            
            tmp["name"] = structname
            
            if charge is not None:
                tmp["charge"] = charge
            
            if len(enthalpy)>0:
                tmp["enthalpy"] = enthalpy[ia]
        
            # add contributing file name
            tmp["files"] = [fd.name.split('/')[-1]]
            structures += [copy.deepcopy(tmp)]

    return structures    
    
def read_castep_geom(fd, index=None, units=units_CODATA2002):
    """Reads a .geom file produced by the CASTEP GeometryOptimization task and
    returns an atoms  object.
    The information about total free energy and forces of each atom for every
    relaxation step will be stored for further analysis especially in a
    single-point calculator.
    Note that everything in the .geom file is in atomic units, which has
    been conversed to commonly used unit angstrom(length) and eV (energy).

    Note that the index argument has no effect as of now.

    Contribution by Wei-Bing Zhang. Thanks!

    Routine now accepts a filedescriptor in order to out-source the *.gz and
    *.bz2 handling to formats.py. Note that there is a fallback routine
    read_geom() that behaves like previous versions did.
    
    ENERGY PARSING NOT SUPPORTED YET
    """    
    flines = fd.readlines()
    # conversion factors from atomic to eV,A,GPa
    factors = {
        't': units['t0'] * 1E15,     # fs
        'E': units['Eh'],            # eV
        'T': units['Eh'] / units['kB'],
        'P': units['Eh'] / units['a0']**3 * units['Pascal']*1E-9, # GPa
        'h': units['a0'],
        'hv': units['a0'] / units['t0'],
        'S': units['Eh'] / units['a0']**3,
        'R': units['a0'],
        'V': np.sqrt(units['Eh'] / units['me']),
        'F': units['Eh'] / units['a0']
    }
        
    # create a list of material properties for each configuration
    energy_total            = []    # 1. total energy
    free_energy_total       = []    # 2. total free energy
    energy_0K               = []    # 3. 0K total energy estimate
    sedc_energy_total       = []    # 4. sedc corrected total energy
    sedc_free_energy_total  = []    # 5. sedc corrected free energy
    sedc_energy_0K          = []    # 6. sedc corrected 0K energy estimate
    MP_energy_total         = []    # 7. Makov-Payne finite basis set corrected total energy
    supercells              = []    # 8. supercell vectors in cartesian
    atoms                   = []    # 9. fractional coordinates of atoms
    species                 = []    # 10. atom types
    forces                  = []    # 11. atom cartesian force components /(eV/A)
    stress                  = []    # 12. stress tensor / (GPa)

    # only take information from the final (converged) configuration
    for ia,line in enumerate(flines):
        if len(line.split())==1:
            readstart = ia
            readend = len(flines)-1
    
    for ia,line in enumerate(flines):
        if ia < readstart or ia > readend:
            continue
        elif ' <-- E\n' in line:
            # convert from Hartree to eV
            #print ('energies: ',float(line.split()[0])*factors['E'], float(line.split()[1])*factors['E'])
            energy_total += [float(line.split()[0])*factors['E']]
        elif ' <-- h\n' in line:
            # 8.
            supercells += [[float(_c)*factors['h'] for _c in line.split()[0:3]]]
        elif ' <-- S\n' in line:
            # 12.
            stress += [[float(_c)*factors['P'] for _c in line.split()[0:3]]]
        elif ' <-- R\n' in line:
            # 9.
            atoms += [[float(_c)*factors['h'] for _c in line.split()[2:5]]]
            # 10.
            species += [line.split()[0]]
        elif ' <-- F\n' in line:
            forces += [[float(_c)*factors['F'] for _c in line.split()[2:5]]]

    # check for .geom corruption
    assert (len(atoms)!=0 and len(species)!=0 and len(supercells)!=0),(fd.name+' appears to be corrupt')
   
    # convert cartesian to fractional coordinates of the cell vectors    
    invcell = np.linalg.inv(np.asarray(supercells))
    for ia in range(len(atoms)):
        r = [_c for _c in atoms[ia][0:3]]
        for ib in range(3):
            tmp = 0
            for ic in range(3):
                tmp += r[ic]*invcell[ic][ib]
            atoms[ia][ib] = tmp

    # set present properties
    geomstruct = supercell()
    info = dict()
    
    if len(forces)==1:
        # atom forces
        info.update({"forces":forces})
    if len(stress)==1:
        # cauchy stress tensor
        info.update({"stress":stress})
    # NEED TO FIGURE OUT WHICH IS 0K ENERGY AND ADD TO CLASS v
    if len(energy_0K)==1:
        print('need to figure what energy is')
    info.update({"cell":supercell,
               "positions":atoms,
               "species":species,
               "files":fd.name.split('/')[-1],
               "name":fd.name.split('/')[-1],})
    # INCONSISTENCIES WITH CASTEP FILE
    #        getattr(geomstruct,geomstruct.set_methods['energy'])(0.0)
            #geomstruct.set_energy(energy_total[0])
    # NEED TO FIGURE OUT WHICH IS 0K ENERGY AND ADD TO CLASS ^


    # Yeah, we know that...
    # print('N.B.: Energy in .geom file is not 0K extrapolated.')
    return [geomstruct(**info)]
    
def read_castep_md(fd, index=None, return_scalars=False,
                   units=units_CODATA2002):
    """Reads a .md file written by a CASTEP MolecularDynamics task
    and returns the trajectory stored therein as a list of atoms object.

    Note that the index argument has no effect as of now.
    
    ENERGY PARSING NOT SUPPORTED YET!!
    """
    flines = fd.readlines()
    factors = {
        't': units['t0'] * 1E15,     # fs
        'E': units['Eh'],            # eV
        'T': units['Eh'] / units['kB'],
        'P': units['Eh'] / units['a0']**3 * units['Pascal']*1E-9, # GPa
        'h': units['a0'],
        'hv': units['a0'] / units['t0'],
        'S': units['Eh'] / units['a0']**3,
        'R': units['a0'],
        'V': np.sqrt(units['Eh'] / units['me']),
        'F': units['Eh'] / units['a0']
    }

    # fd is closed by embracing read() routine
    t0 = time()
    
    num_flines = len(flines)
    print("time reading lines {} s...".format(time()-t0))

    #readstart = [iv+1 for iv in range(num_flines) if 'END header' in flines[iv]][-1]
    readstart = [iv-1 for iv in range(num_flines) if '<-- E' == flines[iv][-6:-1]]
    
    readend = num_flines - 1
    readstart = readstart[1]
    #print("readstart {} readend {}".format(readstart,readend))
    
    # temperorary data
    tmp_energy_total            = []    # 1. total energy
    tmp_free_energy_total       = []    # 2. total free energy
    tmp_energy_0K               = []    # 3. 0K total energy estimate
    tmp_sedc_energy_total       = []    # 4. sedc corrected total energy
    tmp_sedc_free_energy_total  = []    # 5. sedc corrected free energy
    tmp_sedc_energy_0K          = []    # 6. sedc corrected 0K energy estimate
    tmp_MP_energy_total         = []    # 7. Makov-Payne finite basis set corrected total energy
    tmp_supercell               = []    # 8. supercell vectors in cartesian
    tmp_atoms                   = []    # 9. fractional coordinates of atoms
    tmp_species                 = []    # 10. atom types
    tmp_forces                  = []    # 11. atom cartesian force components /(eV/A)
    tmp_stress                  = []    # 12. stress tensor / (GPa)
    energy_total            = []    # 1. total energy
    free_energy_total       = []    # 2. total free energy
    energy_0K               = []    # 3. 0K total energy estimate
    sedc_energy_total       = []    # 4. sedc corrected total energy
    sedc_free_energy_total  = []    # 5. sedc corrected free energy
    sedc_energy_0K          = []    # 6. sedc corrected 0K energy estimate
    MP_energy_total         = []    # 7. Makov-Payne finite basis set corrected total energy
    supercells              = []    # 8. supercell vectors in cartesian
    atoms                   = []    # 9. fractional coordinates of atoms
    species                 = []    # 10. atom types
    forces                  = []    # 11. atom cartesian force components /(eV/A)
    stress                  = []    # 12. stress tensor / (GPa)

    print("time reading lines {} s...".format(time()-t0))
    #for ia,line in enumerate(flines):
    t0 = time()
    #num_energies_read = 0
    for ia in range(num_flines):
        line = flines[ia]
        sline = line.rstrip('\n').split()
        #if ia < 20:
        #    print("line {} sline {}".format(line,sline))
        if ia < readstart or ia > readend:
            continue
            
        elif len(sline)==0 or ia==readend:
            # beginning of another MD step or final MD step
            
            if len(tmp_energy_total)!=0:
                energy_total.append(tmp_energy_total)
            if len(tmp_supercell)!=0:
                supercells.append([ _l for _l in tmp_supercell])
            if len(tmp_atoms)!=0:
                atoms.append([ _a for _a in tmp_atoms])
            if len(tmp_species)!=0:
                species.append([ _s for _s in tmp_species])
            if len(tmp_stress)!=0:
                stress.append([ _s for _s in tmp_stress])
            if len(tmp_forces)!=0:
                forces.append([ _f for _f in tmp_forces])

            # reset tmp data structures
            tmp_energy_total            = []    # 1. total energy
            tmp_free_energy_total       = []    # 2. total free energy
            tmp_energy_0K               = []    # 3. 0K total energy estimate
            tmp_sedc_energy_total       = []    # 4. sedc corrected total energy
            tmp_sedc_free_energy_total  = []    # 5. sedc corrected free energy
            tmp_sedc_energy_0K          = []    # 6. sedc corrected 0K energy estimate
            tmp_MP_energy_total         = []    # 7. Makov-Payne finite basis set corrected total energy
            tmp_supercell               = []    # 8. supercell vectors in cartesian
            tmp_atoms                   = []    # 9. fractional coordinates of atoms
            tmp_species                 = []    # 10. atom types
            tmp_forces                  = []    # 11. atom cartesian force components /(eV/A)
            tmp_stress                  = []    # 12. stress tensor / (GPa)
            #elif ' <-- E\n' in line:
        elif 'E' == sline[-1]:
            # convert from Hartree to eV
            
            tmp_energy_total = [float(sline[0])*factors['E']]
            
            #elif ' <-- h\n' in line:
        elif 'h' == sline[-1]:
            # 8.
            tmp_supercell.append([float(_c)*factors['h'] for _c in sline[0:3]])
            #elif ' <-- S\n' in line:
        elif 'S' == sline[-1]:
            # 12.
            tmp_stress.append([float(_c)*factors['P'] for _c in sline[0:3]])
            #elif ' <-- R\n' in line:
        elif 'R' == sline[-1]:
            # 9.
            tmp_atoms.append([float(_c)*factors['h'] for _c in sline[2:5]])
            # 10.
            tmp_species.append(line.split()[0])
        elif 'F' == sline[-1]:
            tmp_forces.append([float(_c)*factors['F'] for _c in sline[2:5]])
    print("processing md file {} s...".format(time()-t0))
    
    num_frames = len(energy_total)
    print("forces {} supercells {} species {}".format(len(forces),len(supercells),len(species)))
    assert (num_frames>0),"Assertion failed - expected to find at least one timestep, but found {}".format(num_frames)
    
    t0 = time()
    atoms = [np.dot(np.array(atoms[v],dtype=float),np.linalg.inv(np.array(supercells[v],dtype=float))) for v in range(num_frames)]
    print("converting atoms to fspace {} s...".format(time()-t0))
    
    md_structures = [[]]*num_frames

    # parse .md data to in-house format
    t0 = time()
    for ia in range(num_frames):
        info = dict()
        tmp = supercell(fast=True)

        # atom forces
        if len(forces)>0:
            info.update({"forces":forces[ia]})
    
        info.update({"cell":supercells[ia],
               "positions":atoms[ia],
               "species":species[ia],
               "files":[fd.name.split('/')[-1]],
               "name":fd.name.split('/')[-1].split('.')[0]+'-'+str(ia)+'.md',})
                
        tmp(**info)
        md_structures[ia] = tmp
    print("storing md data {} s...".format(time()-t0))
    return md_structures
    
def read_castep_den_fmt(fd,style="array"):
    """
    extract electron density grid points from file fd.

    Parmeters
    ---------
    fd: 
        a non-corrupt .den_fmt castep electron density fil
    style: str
        "array" or "dict", determines the format in which the information is returned.

    Returns
    -------
    list of supercell instance
        [class.supercell()] with density and supercell vector attributes allocated

    """    
    flines = fd.readlines()
    
    # fetch supercell vectors in cartesian coordinates / (A)
    cell = np.array([list(filter(None,v.split(' ')))[:3] for v in flines[3:6]],dtype=float)
    # fetch [N1,N2,N3]; number of grid points along each cell vector
    Npoints = np.array(list(filter(None,flines[8].split(' ')))[:3],dtype=int)
    invNpoints = 1./Npoints
    
    tmp = np.array([list(filter(None,v.split(' '))) for v in flines[12:]],dtype=float)
    points = np.dot((tmp[:,:3]-np.ones(3))*invNpoints,cell)
    density = tmp[:,3]
    
    # create and attribute supercell() structure class
    tmp1 = supercell()
    tmp1["edensity"] = {"xyz":points,"density":density}
    tmp1["cell"] = cell # set supercell vectors
    tmp1["name"] = fd.name.split('/')[-1] # set structure name
    tmp1["files"] = [fd.name.split('/')[-1]] # set file name
    
    return [tmp1]
    
def reduce_castep_den_fmt(fd):
    """Reduce file size by removing entries below 1.0. Store as ascii text characters.

    <fname>_reduced.den_fmt is written from <fname>.den_fmt

    Notes
    -----
    if, instead of ascii text, were to represent single precision floating point in binary
    digits, could further reduce the file size by a factor ~ 2.5. would require a formatted
    binary parser
    """
    
    flines = fd.readlines()

    for ia,line in enumerate(flines):
        if 'END header: data is' in line:
            newflines = ['# modified .den_fmt file format for reduced storage space\n']+\
            flines[:ia+2]

            for ib,_line in enumerate(flines[ia+2:]):
                if float(_line.split()[3]) >= 1.0:
                    newflines += _line

    # write new reduced file
    with open(fd.name.split('.')[0]+'_reduced.den_fmt','w') as f:
        f.writelines(newflines)        


_castep_supported_files = {'castep':read_castep_castep,'md':read_castep_md,'geom':read_castep_geom,\
'den_fmt':read_castep_den_fmt}

class parse:
    """Class to interface castep parsing with general parsing (end user) class

    Supported file types are self.implemented_file_types

    Example
    -------     
    #initialise instance
    castep_parser = parse(/full/dir/path/<file_name>.<file_type> ,file_type=<file_type>)
    
    #parse 
    "/full/dir/path/<file_name>.<file_type>" : castep_parser.run()

    #extract supercell 
    supercell = castep_parser.get_supercells()
    """

    def __init__(self,path,file_type):
        self.implemented_file_types = ['md','castep','geom','den_fmt']

        assert file_type in self.implemented_file_types,'Assertion failed - got unexpected file_type {}. Expected one of: {}'.\
            format(file_type,self.implemented_file_types)

        self.file_type = file_type # file type
        self.path = path # full path to file
        self.supercells = None # supercell

        # internal interface from file type to parser function
        self.internal_interface = {'castep':read_castep_castep,'md':read_castep_md,\
                                   'geom':read_castep_geom,'den_fmt':read_castep_den_fmt}

    def run(self):
        """
        parse self.path
        """
        with open(self.path,'r') as f:
            self.supercells = self.internal_interface[self.file_type](f)
        #assert isinstance(self.supercells,list),'implementation error with parser'

    def get_supercells(self):
        """
        Output:
            - a list of supercell structures
        """
        return self.supercells

if __name__ == "__main__":
    # reduce appropriate den.fmt files in size, in working directory 
    reduce_castep()
   
    # parse supercell structures from all castep affiliated files
    structure_list = parse_castep()
