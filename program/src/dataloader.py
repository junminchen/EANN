import torch
import numpy as np
import torch.distributed as dist

class DataLoader():
    def __init__(self,image,image_A,image_B,image_C,image_D,image_E,image_F,image_G,\
                numatoms,numatoms_A,numatoms_B,numatoms_C,numatoms_D,numatoms_E,numatoms_F,numatoms_G,\
                index_ele,index_ele_A,index_ele_B,index_ele_C,index_ele_D,index_ele_E,index_ele_F,index_ele_G,\
                atom_index,atom_index_A,atom_index_B,atom_index_C,atom_index_D,atom_index_E,atom_index_F,atom_index_G,\
                shifts,shifts_A,shifts_B,shifts_C,shifts_D,shifts_E,shifts_F,shifts_G,\
                label,mol,batchsize,shuffle=True):
        self.image=image
        self.image_A=image_A
        self.image_B=image_B
        self.image_C=image_C
        self.image_D=image_D
        self.image_E=image_E
        self.image_F=image_F
        self.image_G=image_G
        self.numatoms=numatoms
        self.numatoms_A=numatoms_A
        self.numatoms_B=numatoms_B
        self.numatoms_C=numatoms_C
        self.numatoms_D=numatoms_D
        self.numatoms_E=numatoms_E
        self.numatoms_F=numatoms_F
        self.numatoms_G=numatoms_G
        self.index_ele=index_ele
        self.index_ele_A=index_ele_A
        self.index_ele_B=index_ele_B
        self.index_ele_C=index_ele_C
        self.index_ele_D=index_ele_D
        self.index_ele_E=index_ele_E
        self.index_ele_F=index_ele_F
        self.index_ele_G=index_ele_G
        self.atom_index=atom_index
        self.atom_index_A=atom_index_A
        self.atom_index_B=atom_index_B
        self.atom_index_C=atom_index_C
        self.atom_index_D=atom_index_D
        self.atom_index_E=atom_index_E
        self.atom_index_F=atom_index_F
        self.atom_index_G=atom_index_G
        self.shifts=shifts
        self.shifts_A=shifts_A
        self.shifts_B=shifts_B
        self.shifts_C=shifts_C
        self.shifts_D=shifts_D
        self.shifts_E=shifts_E
        self.shifts_F=shifts_F
        self.shifts_G=shifts_G
        self.label=label
        self.mol=mol
        self.batchsize=batchsize
        self.end=self.image.shape[0]
        self.shuffle=shuffle               # to control shuffle the data
        if self.shuffle:
            self.shuffle_list=torch.randperm(self.end)
        else:
            self.shuffle_list=torch.arange(self.end)
        self.length=int(np.ceil(self.end/self.batchsize))
        #print(dist.get_rank(),self.length,self.end)
      
    def __iter__(self):
        self.ipoint = 0
        return self

    def __next__(self):
        if self.ipoint < self.end:
            index_batch=self.shuffle_list[self.ipoint:min(self.end,self.ipoint+self.batchsize)]
            coordinates=self.image.index_select(0,index_batch)
            coordinates_A=self.image_A.index_select(0,index_batch)
            coordinates_B=self.image_B.index_select(0,index_batch)
            coordinates_C=self.image_C.index_select(0,index_batch)
            coordinates_D=self.image_D.index_select(0,index_batch)
            coordinates_E=self.image_E.index_select(0,index_batch)
            coordinates_F=self.image_F.index_select(0,index_batch)
            coordinates_G=self.image_G.index_select(0,index_batch)
            species=self.index_ele.index_select(0,index_batch)
            species_A=self.index_ele_A.index_select(0,index_batch)
            species_B=self.index_ele_B.index_select(0,index_batch)
            species_C=self.index_ele_C.index_select(0,index_batch)
            species_D=self.index_ele_D.index_select(0,index_batch)
            species_E=self.index_ele_E.index_select(0,index_batch)
            species_F=self.index_ele_F.index_select(0,index_batch)
            species_G=self.index_ele_G.index_select(0,index_batch)
            shifts=self.shifts.index_select(0,index_batch)
            shifts_A=self.shifts_A.index_select(0,index_batch)
            shifts_B=self.shifts_B.index_select(0,index_batch)
            shifts_C=self.shifts_C.index_select(0,index_batch)
            shifts_D=self.shifts_D.index_select(0,index_batch)
            shifts_E=self.shifts_E.index_select(0,index_batch)
            shifts_F=self.shifts_F.index_select(0,index_batch)
            shifts_G=self.shifts_G.index_select(0,index_batch)
            numatoms=self.numatoms.index_select(0,index_batch)
            numatoms_A=self.numatoms_A.index_select(0,index_batch)
            numatoms_B=self.numatoms_B.index_select(0,index_batch)
            numatoms_C=self.numatoms_C.index_select(0,index_batch)
            numatoms_D=self.numatoms_D.index_select(0,index_batch)
            numatoms_E=self.numatoms_E.index_select(0,index_batch)
            numatoms_F=self.numatoms_F.index_select(0,index_batch)
            numatoms_G=self.numatoms_G.index_select(0,index_batch)
            atom_index=self.atom_index[:,index_batch]
            atom_index_A=self.atom_index_A[:,index_batch]
            atom_index_B=self.atom_index_B[:,index_batch]
            atom_index_C=self.atom_index_C[:,index_batch]
            atom_index_D=self.atom_index_D[:,index_batch]
            atom_index_E=self.atom_index_E[:,index_batch]
            atom_index_F=self.atom_index_F[:,index_batch]
            atom_index_G=self.atom_index_G[:,index_batch]
            abprop=(label.index_select(0,index_batch) for label in self.label)
            mol=self.mol.index_select(0,index_batch)
            self.ipoint+=self.batchsize
            #print(dist.get_rank(),self.ipoint,self.batchsize)
            return abprop,mol,coordinates,coordinates_A,coordinates_B,coordinates_C,coordinates_D,coordinates_E,coordinates_F,coordinates_G,\
                    numatoms,numatoms_A,numatoms_B,numatoms_C,numatoms_D,numatoms_E,numatoms_F,numatoms_G,\
                    species,species_A,species_B,species_C,species_D,species_E,species_F,species_G,\
                    atom_index,atom_index_A,atom_index_B,atom_index_C,atom_index_D,atom_index_E,atom_index_F,atom_index_G,\
                    shifts,shifts_A,shifts_B,shifts_C,shifts_D,shifts_E,shifts_F,shifts_G
        else:
            # if shuffle==True: shuffle the data 
            if self.shuffle:
                self.shuffle_list=torch.randperm(self.end)
            #print(dist.get_rank(),"hello")
            raise StopIteration
