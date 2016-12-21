#!/bin/env python
"""
    Usage:
        pyneb.py 
        pyneb.py <N> [-n <calc-name>] [-o <optimiser>] [-i <initial-guess>]

An interface to ase's nudged elastic band (NEB) implementation. Creates .cell 
and .param files for all images and each iteration of the band. CASTEP 
singlepoint calculations must be performed on each generated structure, before
running pyneb again.

Annoyingly ASE only supports fixed cell NEB.

Method
------
1. Generate two seed files (the end points of the band, which are assumed to be 
   (meta)stable) and include these .cell and .param files for the desired 
   singlepoint CASTEP calculation in the working directory.

2. Run pyneb specifying <N>, the number of desired images, along with all other 
   optional arguments listed below.

>>> pyneb.py 7 -n phase_change_1 -o MDmin -i linear

3. Run CASTEP on all <calc-name>_1-i.(cell/param) files generated, i=[1,<N>]

4. Once calculations are finished, run pyneb to generate <calc-name>_2-i.

>>> pyneb.py

5. Repeat steps 3. and 4. until user is content.



Arguments:
    <N>                 The number of images to use for the calculation. <N>
                        must be greater than two.

Optional:
    -n <calc-name>      The name given to castep files. Default is 'neb-calc'
    -o <optimiser>      The method used to minimise the NEB hamiltonian. 
                        Supported choices are:
                            - BFGS   (default)
                            - LBFGS  (untested)
                            - MDmin  (untested)
                            - FIRE   (untested)
    -i <initial-guess>  The method used to interpolate an initial guess for the
                        NEB path. Supported choices are:
                            - IDPP   (default)
                            - linear 
"""

#-------------------------------------------#
# standard modules, with modified ase.atoms #
#-------------------------------------------#

from docopt import docopt
from os import listdir,mkdir
from os import path as ospat
from os import path as ospath
from sys import version_info,exit,path
from copy import deepcopy
from shutil import copyfile
from ase.neb import NEB
from ase.calculators.emt import EMT
from ase.optimize import BFGS,LBFGS,MDMin,FIRE

#-----------------------------------------------------#
# append location of pyneb library files, to sys.path #
#-----------------------------------------------------#
pyneb_lib = '/home/atf29/Documents/bitbucket_local/pyneb'

path.append(pyneb_lib)

#---------------#
# pyneb modules #
#---------------#

from pyneb_parsecell import fetchkeywords as fetchkeywords
from pyneb_parsecell import asetocell as asetocell
from pyneb_parsecell import comparekeywords as comparekeywords
from pyneb_parsecell import celltoase as celltoase
from pyneb_parsecell import checkparamfiles as checkparamfiles
from pyneb_parsecell import adoptcellorder as adoptcellorder
from pyneb_parsecell import checkimages as checkimages
from pyneb_parser_general import GeneralInputParser as GeneralInputParser


# format library destination to end with '/'
if pyneb_lib.split('/')[-1]!='':
    pyneb_lib+='/'

if __name__ == "__main__":
    # check user is running on python3
    assert version_info.major==3,"Script requires Python 3, you're running on {}.{}".\
    format(version_info.major,version_info.minor)

    args = docopt(__doc__,version=1.0)
    
    # supported NEB hamiltonian minimisation methods
    supp_opt_methods = ['bfgs','lbfgs','mdmin','fire']

    # supported initial math estimation methods
    supp_ini_methods = ['linear','idpp']

    # file for NEB metadata
    optfile ='neb_metadata'

    # dir name for NEB files
    nebdir = 'pyneb-internal-files/'

    #----------------------#
    # command line parsing #
    #----------------------#
    
    if args["<N>"]:
        Nim = int(args["<N>"])
        Nbands = 0
    else:
        # list files in wrkdir to get Nim
        files = listdir('.')

        # sets to count present images and bands
        imge_cntr = set([])
        band_cntr = set([])

        for _f in files:
            if '.cell' in _f.lower():
                if len(_f.split('.')[0].split('_'))>1 and \
                len(_f.split('.')[0].split('_')[-1].split('-'))>1:
                    # assume <calc-name>_i-j.cell formatting where i,j are band, image numbers
                    band_cntr.update(set([int(_f.split('.')[0].split('_')[-1].split('-')[0])]))
                    imge_cntr.update(set([int(_f.split('.')[0].split('_')[-1].split('-')[1])]))

                    # system name
                    sys_name = ''.join(_f.split('.')[0].split('_')[0:-1])

        if len(imge_cntr)==0:
            msg = 'You need to initialise an initial band and do some DFT before trying to '
            msg += 'generate the next band. See documentation: \n'
            msg += '>>> pyneb.py -h'
            print (msg)
            exit()
        else:
            # number of images
            Nim = max(imge_cntr)

            # number of bands
            Nbands = max(band_cntr)

    # need to store optimiser details in file ! 

    if args["-i"]:
        assert args["-i"].lower() in supp_ini_methods,\
        '{} is not a supported initial guess method: {}'.format(args["-i"],supp_ini_methods)

        interp_method = args["-i"].lower()
    else:
        interp_method = 'idpp'

    if args["-o"]:
        assert args["-o"].lower() in supp_opt_methods,\
        '{} is not a supported NEB optimisation method: {}'.format(args["-o"],supp_opt_methods)

        opt_method = args["-o"].lower()
    elif args["<N>"]:
        opt_method = 'bfgs'
    else:
        # check that nebdirectory exists
        assert ospath.isdir(nebdir),'Data has become corrupted: {} is not present in working directory.'\
        .format(nebdir)

        # need to read optimisation method from file
        with open(nebdir+optfile,'r') as f:
            flines = f.readlines()

        # if true, error reading file containing optimisation method
        readerror = [True,True]

        for l in flines:
            if 'NEB hamiltonian minimisation method :' in l:
                opt_method = l.split(':')[-1].split()[0]
            
                if opt_method in supp_opt_methods:
                    readerror[0] = False
            elif 'NEB initial band interpolation method' in l:
                interp_method = l.split(':')[-1].split()[0]

                if interp_method in supp_ini_methods:
                    readerror[1] = False
        if any(readerror):
            raise ValueError('Error - NEB metadata file {} has been corrupted!'.format(optfile))
    
    # if initial run, save NEB metadata to optfile
    if args["<N>"]:
        if ospath.isdir(nebdir):
            print ('pyneb has already been run in this directory. If you want to start again, remove {} first'.\
            format(nebdir))
        else:
            # first, create neb_dir
            mkdir(nebdir)

        with open(nebdir+optfile,'w') as f:
            flines = ["##### NEB metadata file - it's formatted so touch at your own peril #####\n\n",\
            'NEB hamiltonian minimisation method : '+opt_method+'\n',\
            'NEB initial band interpolation method : '+interp_method]

            f.writelines(flines)

    if args["-n"]:
        sys_name = args["-n"]
    elif args["<N>"]:
        sys_name = 'neb-calc'

    # check at least 3 images are specified
    assert Nim>=3,'{} images is an unsuitable number for NEB'.format(Nim)


    # subsidiary routines
    def printheader(sys_name,opt_method,interp_method,Nim,Nbands,progress,energies,Nconst,nebdir):
        # number of generated bands
        if len(energies)==Nbands:
            genbands = Nbands+1
        else:
            genbands = Nbands


        # list of lines for output to stdout and file
        flines = {'stdout':[],'file':[]}

        flines['stdout'].append('-------------------------------------------------------------------------')
        flines['stdout'].append('================================= pyneb =================================')
        flines['stdout'].append('-------------------------------------------------------------------------\n')
        flines['stdout'].append('===========')
        flines['stdout'].append('Run summary')
        flines['stdout'].append('===========\n')
        flines['stdout'].append('{:<40}'.format('System name')+' : '+sys_name)
        flines['stdout'].append('{:<40}'.format('Number of images per band')+' : '+str(Nim))
        flines['stdout'].append('{:<40}'.format('NEB hamiltonian minimiser')+' : '+opt_method)
        flines['stdout'].append('{:<40}'.format('Initial band interpolation method')+' : '+interp_method)
        flines['stdout'].append('{:<40}'.format('Fixed cell vectors')+' : true')
        flines['stdout'].append('{:<40}'.format('Number of cartesian atomic constraints')+' : '+str(Nconst))
        flines['stdout'].append('{:<40}'.format('Bands generated so far')+' : '+str(genbands))
        flines['stdout'].append('{:<40}'.format('Bands on which DFT has been completed')+' : '+str(len(energies))+'\n')
       
        # fetch NEB energies
        NEB_energies = []
       
        if Nbands!=0:
            with open(nebdir+sys_name+'.log','r') as f:
                flines2 = f.readlines()

                for _line in flines2:
                    NEB_energies.append(float(_line.split()[3]))

        assert len(energies)==len(NEB_energies),'{} file possibly corrupted'.format('.log')

        if len(energies)!=0:
            flines['stdout'].append('===========')
            flines['stdout'].append('NEB summary')
            flines['stdout'].append('===========\n')

        for i in range(len(energies)):
            flines['stdout'].append('------')
            flines['stdout'].append('Band '+str(i+1))
            flines['stdout'].append('------\n')
            flines['stdout'].append('band number    image number    file name                      energy / (eV)')
            for j,_file in enumerate(sorted(energies[i])):
                flines['stdout'].append('{:<3}            {:<4}            {:<30} {}'.\
                format(str(i+1),str(j+1),_file,energies[i][_file]))
            flines['stdout'].append('\nband energy : {}'.format(NEB_energies[i]))
            flines['stdout'].append('')

        for _l in flines['stdout']:
            flines['file'].append(_l+'\n')

        # print information about status of latest CASTEP dft runs
        if len(progress)!=0:
            flines['stdout'].append('Progress of DFT for most recent band (band '+str(Nbands)+')\n')
            flines['file'].append(flines['stdout'][-1]+'\n')
            
            for _file in sorted(progress):
                if progress[_file]:
                    flines['stdout'].append('\033[1;32m{}\033[1;m'.format(_file))
                    flines['file'].append('{:<40} (complete)\n'.format(_file))
                else:
                    flines['stdout'].append('\033[1;31m{}\033[1;m'.format(_file))
                    flines['file'].append('{:<40} (incomplete)\n'.format(_file))
            
            if not all([progress[_f] for _f in progress]):
                flines['stdout'].append('\nDFT calculations have not finished on band {}, cannot iterate to next band yet'.\
                format(Nbands))
                flines['file'].append(flines['stdout'][-1]+'\n')
            else:
                flines['stdout'].append('\nDFT calculations for band {} are complete, have generated {} new images for band {}\n'.\
                format(Nbands,Nim-2,Nbands+1))
                flines['file'].append(flines['stdout'][-1]+'\n')

                for i in range(1,Nim+1):
                    _file = sys_name+'_'+str(Nbands+1)+'-'+str(i)+'.castep'
                    if i==1 or i==Nim:
                        flines['stdout'].append('\033[1;32m{}\033[1;m'.format(_file))
                        flines['file'].append('{:<40} (complete)\n'.format(_file))
                    else:
                        flines['stdout'].append('\033[1;31m{}\033[1;m'.format(_file))
                        flines['file'].append('{:<40} (incomplete)\n'.format(_file))

            # write file
            with open('summary.pyneb','w') as f:
                f.writelines(flines['file'])

            # print to stdout
            for _l in flines['stdout']:
                print (_l)

    if args["<N>"]:
        #-----------------------#
        # initialise NEB images #
        #-----------------------#

        #-------------------------------------#
        # check .cell and .param files are OK #
        #-------------------------------------#

        # check that only 2 cell files are present in the working directory
        files = listdir('.')

        initial_cell = []
        initial_param = [None,None]

        for _f in files:
            if '.cell' in _f.lower():
                initial_cell.append(_f)

        msg = 'cell file(s) in the working directory. Must have exactly 2 seed files '
        msg += 'in the working directory when creating the first band of images. See documentation:\n'
        msg += '>>> pyneb.py -h'

        assert len(initial_cell)==2,'Have found {} '.format(len(initial_cell))+msg


        msg = 'Cannot find .param files in the working directory for the seed .cell files'

        # check for associated .param files
        assert all([initial_cell[i].split('.')[0]+'.param' in files for i in range(2)]),msg

        for _f in files:
            if '.param' in _f.lower():
                for i in range(2):
                    if initial_cell[i].split('.')[0] in _f:
                        initial_param[i] = _f
                    
        # check .param files are identical and singlepoints outputing forces
        checkparamfiles(initial_param[0],initial_param[1])
        
        # check keywords in both seed cell files are the same
        keywords = [fetchkeywords(initial_cell[0]),fetchkeywords(initial_cell[1])]

        comparekeywords(keywords)

        #------------------------------------------#
        # read in .cell files as ase Atoms objects #
        #------------------------------------------#
        
        images = [celltoase(initial_cell[0])]
        
        for i in range(1,Nim-1):
            images.append(deepcopy(images[0]))
        images.append(celltoase(initial_cell[1]))

        # check that seed structures are actually different!
        checkimages(images[0],images[-1])

        # create NEB object for using clever initial interpolation
        neb = NEB(images)

        if interp_method == 'linear':
            # must have mic=True for pbcs to be accounted for
            neb.interpolate(mic=True)
        else:
            # must have mic=True for pbcs to be accounted for
            neb.interpolate('idpp',mic=True)
        
    
        # check cell is fixed
        assert all([images[0].cell[i][j] == images[-1].cell[i][j] for i in range(3) for j in range(3)]),\
        'initial and final cells are not identical. Internal ASE only supported fixed cell NEB, sorry.'


        # check keywords in both seed cell files are the same
        keywords = [fetchkeywords(initial_cell[0]),fetchkeywords(initial_cell[1])]

        comparekeywords(keywords)

        # output ase Atoms objects to .cell files and copy .param

        for i in range(1,Nim+1):
            # <sys_name>_1-i.cell file name
            _cellfile = sys_name+'_1-'+str(i)+'.cell'
            _parmfile = sys_name+'_1-'+str(i)+'.param'

            # parse ase Atoms object to a castep cell file 
            asetocell(neb.images[i-1],keywords[0],_cellfile)

            # copy initial param file
            copyfile(initial_param[0],_parmfile)

        # dummy variables
        progress = {}
        energies = []

    else:
        #-----------------------#
        # perform next NEB step #
        #-----------------------#

        files = listdir('.')

        #---------------------------------------------#
        # reread all DFT energies from previous bands #
        #---------------------------------------------#

        # MUST use list comprehension, [{}]*(Nbands-1) creates only ONE dict!
        energies = [{} for i in range(Nbands-1)]

        for i in range(Nbands-1):
            # loop over all previous bands before the most recent

            for j in range(Nim):
                _file = sys_name+'_'+str(i+1)+'-'+str(j+1)+'.castep'

                assert _file in files,'NEB working directory has become corrupted, {} is not present'.\
                format(_file)

                parser = GeneralInputParser()

                parser.parse_file(_file)
                
                energies[i].update({_file:parser.supercells[0].energy})

                del parser

        progress = {}
    
        #----------------------------------#
        # check to see if DFT has finished #
        #----------------------------------#

        for i in range(Nim):
            tmp = True
            
            _file = sys_name+'_'+str(Nbands)+'-'+str(i+1)+'.castep'

            if _file not in files:
                # check if .castep file has been written
                tmp = False
            else:
                # if written, check if calculation has finished
                with open(_file,'r') as f:
                    flines = f.readlines()

                    # check for 'Calculation time' 
                    if not any(['Calculation time' in l for l in flines]):
                        # phrase not found, calculation incomplete
                        tmp = False
                    
                    del flines
            
            progress.update({_file:tmp})

        # check if all calculations are finished
        if all([progress[_file] for _file in progress]):

            # if here, then all castep files from most recent band exist
            tmpdict = {}
            
            images = []
            
            for i in range(Nim):
                _file = sys_name+'_'+str(Nbands)+'-'+str(i+1)+'.castep'
                
                parser = GeneralInputParser()

                parser.parse_file(_file)

                tmpdict.update({_file:parser.supercells[0].energy})
              
                cstp_atm = parser.supercells[0].ase_format()
                cell_atm = celltoase(sys_name+'_'+str(Nbands)+'-'+str(i+1)+'.cell')
                
                # reorder atoms to .cell order and include atom constraints
                images.append(adoptcellorder(cstp_atm,cell_atm))
                 
                # deprecated
                #images.append(parser.supercells[0].ase_format())

                del parser
            energies.append(tmpdict)

            #--------------------#
            # generate next band #
            #--------------------#


            # create NEB object
            neb = NEB(images,method='improvedtangent')

            restart_name = nebdir+sys_name+'.pckl'
            traject_name = nebdir+sys_name+'.traj'
            logfile_name = nebdir+sys_name+'.log'
            #traject_name = nebdir+sys_name+'_'+str(Nbands)+'.traj'

            # initialise optimistation object
            if opt_method == 'bfgs':
                qn = BFGS(neb,restart=restart_name,trajectory=traject_name,logfile=logfile_name)
            elif opt_method == 'lbfgs':
                qn = LBFGS(neb,restart=restart_name,trajectory=traject_name,logfile=logfile_name)
            elif opt_method == 'mdmin':
                qn = MDMin(neb,restart=restart_name,trajectory=traject_name,logfile=logfile_name)
            elif opt_method == 'fire':
                qn = FIRE(neb,restart=restart_name,trajectory=traject_name,logfile=logfile_name)

            # perform 1 step, generating next iteration
            qn.run(fmax=0.05,steps=1,castep_neb=True)

           
            # cell file keywords from previous band
            keywords = fetchkeywords(sys_name+'_'+str(Nbands)+'-1.cell')

            # output new band of images
            for i,img in enumerate(neb.images):
                # file name
                _file = sys_name+'_'+str(Nbands+1)+'-'+str(i+1)+'.cell' 
            
                # output to .cell
                asetocell(img,keywords,_file)

                # output to .param
                copyfile(sys_name+'_'+str(Nbands)+'-1.param',_file.split('.')[0]+'.param')
        
            # copy pckl file and traj file
            copyfile(restart_name,nebdir+sys_name+'_'+str(Nbands)+'.pckl')
            copyfile(traject_name,nebdir+sys_name+'_'+str(Nbands)+'.traj')

            # copy .castep file for end points from previous band
            copyfile(sys_name+'_'+str(Nbands)+'-1.castep',sys_name+'_'+str(Nbands+1)+'-1.castep')
            copyfile(sys_name+'_'+str(Nbands)+'-'+str(Nim)+'.castep',\
            sys_name+'_'+str(Nbands+1)+'-'+str(Nim)+'.castep')
        else:
            # DFT calculations of current band are not complete
            
            # dummy .cell file from current band to fetch number of constraints
            images = [celltoase(sys_name+'_'+str(Nbands)+'-1.cell')]*3
            
            # dummy neb object for pyneb printing to stdout
            neb = NEB(images)


    # print info to stdout
    printheader(sys_name,opt_method,interp_method,Nim,Nbands,progress,energies,\
    len(neb.images[0].constraints),nebdir)
