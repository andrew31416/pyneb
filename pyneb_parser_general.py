import sys, os, copy, pickle
import numpy as np
import pandas as pd
import pyneb_parser_castep
from pyneb_parser_castep import deprecate
import pyneb_structureformat
from time import time

def write_supercells2pickle(path,supercells):
    # write data from supercell form into a dataframe and write that to disk as a csv file
    print("Writing pickle to {}...".format(path))
    with open(path,'wb') as f:
        pickle.dump(supercells,f)

def read_pickle2supercells(path):
    with open(path,'rb') as f:
        supercells = pickle.load(f)
    
    return supercells

class GeneralInputParser:
    """General parser.
    
    General parser for the end-user. This parser calls DFT code specific parsers
    which have standardized attribute names: "energy", "force", "xyz", "cell" and "species".
    The specific parsers can be called via PARSERNAME(path,file_type='FILETYPE').run().
    
    
    Attributes
    ----------
        
    implemented_file_types : dict
        a dictionary of supported file extensions with the 
        associated dft code

    implemented_dft_codes  : dict
        a dictionary of dft codes and an interface to the
        associated parser. dft code parsers must accept 2 arguments:
        (path,file_type) , where path is a single file given by its
        full path and file_type is the file extension as declared in
        self.implemented_file_types

    parse_file(path) : str
        parse a specific file, 'path', assuming the file extension 
        is recognised in self.implemented_file_types

    parse_all(path) : 
        parse all files in 'path' (a string or list of strings), 
        giving the target directory(ies), the extensions of which 
        are recognised in self.implemented_file_types
    
    Notes
    -----
    This class assumes that each DFT calculation is represented with as
    unique prefix in its name (i.e. structure["name"].split('.')[0] if structure
    is an instance of the class supercell). all files with the same prefix will
    merged.
    This convention may be awkward at first for VASP but should in the end be
    usful for bookkeeping.
    """
    
    def __init__(self):
        # supported file types and associated dft code
        self.implemented_file_types = {'xml':'vasp','contcar':'vasp',\
        'md':'castep','geom':'castep','castep':'castep','den_fmt':'castep'}
        
        # interface to dft code parsers
        #self.implemented_dft_codes = {'vasp':parser_vasp.parse,'castep':parser_castep.parse}
        self.implemented_dft_codes = {'castep':pyneb_parser_castep.parse}
        #self.implemented_file_types = {'vasp':['vasprun','contcar'],'castep':['md','geom','castep','den_fmt']}
       
        # file extensions must be unique, search for duplicates
        for ia,_typea in enumerate(self.implemented_file_types):
            for ib,_typeb in enumerate(self.implemented_file_types):
                if ia!=ib and _typea==_typeb:
                    raise ValueError('implementation error, duplicate file types: {} {}'.format(_typea,typeb))

        # list of supercell class objects
        self.supercells = None 
        
    def parse_file(self,path):
        """
        parse a single file, given that the file exists and has a file extension
        consistent with the supported file types.

        Input:

            - path : the full path of a file intended to be parsed

        Output:
            - self.supercells : append path's structure to the current list of
                                structurs in self.supercells
        """

        assert os.path.exists(path),'{} does not exist'.format(path)
        # assume unix system
        assert (len(path.split('/')[-1].split('.'))==2 and \
        path.split('/')[-1].split('.')[-1].lower() in self.implemented_file_types),\
        '{} is not a supported file type: {}'.format(path,[_a for _a in self.implemented_file_types])
        
        # file type
        file_extension = path.split('/')[-1].split('.')[-1].lower()

        """
        pass file's path and extension to appropriate code's parser

        perform internal check that file type is supported and interface to 3rd 
        party parsing codes if appropriate.
        """
        parser_obj = self.implemented_dft_codes[self.implemented_file_types[file_extension]]\
        (path,file_type=file_extension)

        # parse the file 'path'
        parser_obj.run()

        assert isinstance(parser_obj.supercells,list),'implementation error, parser must return a list'

        # append the parsed structure to self.supercells
        if isinstance(self.supercells,type(None)):
            self.supercells = []
        self.supercells += copy.deepcopy(parser_obj.supercells)
       
        del parser_obj

    def parse_all(self,path):
        """
        parse the structure information from all supported file types in 'path', 
        a directory or list of directories

        Parameters
        ----------    
        path : str or list of str
            a string or list of strings giving all directories from which to 
            parse all supported file types in self.implemented_file_types
        
        Returns
        -------
        self.supercells : 
            append acquired structures to self.supercells, a list of all parsed structures
                                
        Notes
        -----
        Important: When setting the name of the supercell make sure its prefix 
        is unique when reading multiple files such that each complete structure
        can be constructed from all supercells with identical prefix and varying suffix.
        """

        if isinstance(path,str):
            directory_list = [path]
        elif isinstance(path,list):
            assert all(isinstance(_a,str) for _a in path),'Assertion failed - expected a list of strings, got {} instead.'.format(path)
            directory_list = copy.deepcopy(path)
        else:
            raise ValueError("Paremeter 'path' ({}) must be a string or list of strings".format(path))
            
        # loop over all target directories
        t0 = time()
        for ia,_dir in enumerate(directory_list):
            # ensure directory path finishes with "/" - assume unix
            if _dir[len(_dir)-1]!='/':
                _dir += '/'

            # check _dir exists!
            assert os.path.isdir(_dir),"Assertion failed - specified path ({}) is not a directory!".format(_dir)

            # list of files in _dir
            tmpfiles = [_f for _f in os.listdir(_dir) if os.path.isfile(os.path.join(_dir,_f))]
    
            # include only files with a delimator
            files = [_f for _f in tmpfiles if len(_f.split('.'))>=2]

            # check for anomylous delimators
            #assert all(len(_f.split('.'))==2 for _f in files),'all files must have a single "." delimator'

            files = sorted(files)
            for _f in files:
                file_extension = _f.split('.')[-1]
                print(">file {}".format(_f))
                if file_extension in self.implemented_file_types:
                    # create instance of dft code parser for _f
                    parser_obj = self.implemented_dft_codes[self.implemented_file_types[file_extension]]\
                        (directory_list[ia]+_f,file_type=file_extension)

                    # parse _f
                    parser_obj.run()
        
                    #assert isinstance(parser_obj.supercells,list),\
                    #    'implementation error, parser must return a list'

                    # append to self.supercells
                    if self.supercells is None:
                        self.supercells = []
                    self.supercells.extend(copy.deepcopy(parser_obj.supercells))
                    
        print("looping all directories {} s...".format(time()-t0))
        # search for and merge segments of the same structure, overwriting self.supercells
        t0 = time()
        self.merge_supercells()
        print("merging supercells {} s...".format(time()-t0))

    def get_supercells(self):
        return self.supercells
                        
    def merge_supercells(self,new=True): #ONLY SUPPORTS CASTEP!
        """
        determine if any structures in self.supercells are duplicates, or if a number
        of structures contains segments of information from the same structure.

        merge any segmented or duplicte structures into a new distint supercell object, 
        overwriting the self.supercells from file reading.
        
        Components:
            - merge_group(group,structure_name) : merges a list of supercell objects 'group'
                                                  into a single object of name 'structure_name'.
            
            - sort_groups(structures)           : sort a list of structure objects, 'structures'
                                                  into a list of lists of dictionaries containing
                                                  the element index and object name in 'structures'
                                                  that belong to a given group.
                - unequal(float1,float2)        : return true if float1 != float2, false otherwise
        """
        
        assert isinstance(getattr(self,'supercells'),list),"Assertion failed - expected the attribute to be a list, got {} instead!".format(type(self.supercells))
        
        # sort structure objects into distinct structures
        def _merge_castep(name,idx_dict,keys): 
            s = pyneb_structureformat.supercell(fast=False)
            s["name"] = name
            file_order = {"castep":0,"den_fmt":1,"md":2}
            
            idx_sorted = sorted(idx_dict[name], key = lambda x: file_order[self.supercells[x]["name"].split('.')[-1]])
            
            for ix in idx_sorted:
                for key in keys:
                    value = self.supercells[ix][key]
                    if value is not None:
                        if key == "files":
                            s["files"].append(value)
                        if key == "energy":
                            if self.supercells[ix]["name"].split('.')[-1]=="castep":
                                try: #writing only those values to supercell which are not yet known
                                    s[key] = value
                                except:
                                    pass 
                        else:
                            try:
                                s[key] = value
                            except:
                                pass
            return s
        
        merge_funs = {"castep":_merge_castep,}
        
        if new:
            num_scells = len(self.supercells)
            unique_prefixes = ['.'.join(s["name"].split('.')[:-1]) for s in self.supercells]
            idx_dict = dict()
            for i in range(num_scells):
                name = unique_prefixes[i]
                
                #group supercell objects by name prefix 
                if name in idx_dict:
                    idx_dict[name].append(i)
                else:
                    idx_dict[name] = [i]
                
            unique_prefixes = set(unique_prefixes)
            unique_structures = {name:None for name in unique_prefixes}
            keys = set(self.supercells[0].keys()).difference(["name","files"])
            unique_prefixes = sorted(list(unique_prefixes))
            
            for i,name in enumerate(unique_prefixes):
                #this is a bit indirect but does: get the suffix of the first listed file associated with the unique prefix and fetches the dft code
                #this is then used to merge the data from the associated files appropriately 
                dft_code_suffix = self.supercells[idx_dict[name][0]]["name"].split('.')[-1]
                unique_structures[name] = merge_funs[self.implemented_file_types[dft_code_suffix]](name,idx_dict,keys)
                
            #store the merged supercells removing all without energy values
            self.supercells = [unique_structures[val] for val in unique_prefixes if unique_structures[val]["energy"] is not None]
            
        else:
            import gc

            groups = sort_groups(self.supercells)

            # remove duplicate groups
            glist = []

            for ia,_g in enumerate(groups):
                # produce list of lists of structure indices
                glist += [[[___g for ___g in __g][0] for __g in _g]]
                glist[ia].sort()

            groups = []

            # form list of distinct lists of structures indices
            for ia,_group in enumerate(glist):
                if _group not in groups:
                    groups += [_group]

            # form list of groups (a list) of unique structures 
            unique_structures = []

            # query all structure names on the stack and carry on from laste "structure_x" x value
            stack_names = []
            for obj in gc.get_objects():
                if (isinstance(obj,pyneb_structureformat.supercell)):
                    # check for supercells created by merge_supercells()
                    if getattr(obj,obj.get_methods['name'])().split('_')[0]=='structure':
                        stack_names.append(getattr(obj,obj.get_methods['name'])())
            # if first time merge_supercells() has been called
            if len(stack_names)==0:
                xval = 0
            else:
                xval = max([int(_name.split('_')[1]) for _name in stack_names])

            for ia in range(len(groups)):
                tmp = []
                for _g in groups[ia]:
                    tmp += [self.supercells[_g]]
                unique_structures += [merge_group(tmp,'structure_'+str(xval+ia+1))]
                del tmp
            # overwrite old file structures from stack
            self.supercells = copy.deepcopy(unique_structures)

            # free unused memory
            del unique_structures
        
