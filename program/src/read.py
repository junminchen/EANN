import os
import gc
import torch
import numpy as np
from src.read_data import *
from src.get_info_of_rank import *
from src.gpu_sel import *
# used for DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# open a file for output information in iterations
fout=open('nn.err','w')
fout.write("EANN Package used for fitting energy and tensorial Property\n")

# global parameters for input_nn
start_table=0                  # 0 for energy 1 for force 2 for dipole 3 for transition dipole moment 4 for polarizability
table_coor=0                   # 0: cartestion coordinates used 1: fraction coordinates used
table_init=0                   # 1: a pretrained or restart  
nblock = 1                     # nblock>=2  resduial NN block will be employed nblock=1: simple feedforward nn
ratio=0.9                      # ratio for vaildation
#==========================================================
Epoch=10000                  # total numbers of epochs for fitting 
patience_epoch=500              # patience epoch  Number of epochs with no improvement after which learning rate will be reduced. 
decay_factor=0.6               # Factor by which the learning rate will be reduced. new_lr = lr * factor.      
print_epoch=10                 # number of epoch to calculate and print the error
# adam parameter                 
start_lr=0.001                  # initial learning rate
end_lr=1e-4                    # final learning rate
#==========================================================
# regularization coefficence
re_ceff=0.0                 # L2 normalization cofficient
batchsize=32                  # batch size 
e_ceff=1.0                    # weight of energy
init_f = 5                 # initial force weight in loss function
final_f = 5e-1                # final force weight in loss function
nl=[256,128,64,32]                  # NN structure
dropout_p=[0.0,0.0,0.0]           # dropout probability for each hidden layer
activate = 'Relu_like'          # default "Tanh_like", optional "Relu_like"
queue_size=10
table_norm= False
find_unused = False
DDP_backend="nccl"
# floder to save the data
floder="./"
dtype='float64'   #float32/float64
#======================read input_nn=================================================================
with open('para/input_nn','r') as f1:
   while True:
      tmp=f1.readline()
      if not tmp: break
      string=tmp.strip()
      if len(string)!=0:
          if string[0]=='#':
              pass
          else:
              m=string.split('#')
              exec(m[0])

if dtype=='float64':
    torch_dtype=torch.float64
    np_dtype=np.float64
else:
    torch_dtype=torch.float32
    np_dtype=np.float32

# set the default type as double
torch.set_default_dtype(torch_dtype)

#======================read input_density=============================================
# defalut values in input_density
nipsin=2
cutoff=6.0
nwave=12
with open('para/input_density','r') as f1:
   while True:
      tmp=f1.readline()
      if not tmp: break
      string=tmp.strip()
      if len(string)!=0:
          if string[0]=='#':
             pass
          else:
             m=string.split('#')
             exec(m[0])

#================ end of read parameter from input file================================

# define the outputneuron of NN
if start_table<=2:
   outputneuron=1
elif start_table==3:
   outputneuron=3
elif start_table==4:
   outputneuron=1

#========================use for read rs/inta or generate rs/inta================
maxnumtype=len(atomtype)
if 'rs' in locals().keys():
   rs=torch.from_numpy(np.array(rs,dtype=np_dtype))
   inta=torch.from_numpy(np.array(inta,dtype=np_dtype))
   nwave=rs.shape[1]
else:
   inta=torch.ones((maxnumtype,nwave))*0.4
   rs=torch.stack([torch.linspace(0,cutoff,nwave) for itype in range(maxnumtype)],dim=0)

# increase the nipsin
nipsin+=1
norbit=nipsin*nwave
nl.insert(0,norbit)

#=============================================================================
floder_train=floder+"train/"
floder_test=floder+"test/"
# obtain the number of system
floderlist=[floder_train,floder_test]
# read the configurations and physical properties
if start_table==0 or start_table==1:
    numpoint,atom,mass,numatoms,scalmatrix,period_table,coor,pot,force,mol=  \
    Read_data(floderlist,1,start_table=start_table)
elif start_table==2 or start_table==3:
    numpoint,atom,mass,numatoms,scalmatrix,period_table,coor,dip,force=  \
    Read_data(floderlist,3)
else:
    numpoint,atom,mass,numatoms,scalmatrix,period_table,coor,pol,force=  \
    Read_data(floderlist,9)

#============================convert form the list to torch.tensor=========================
numpoint=np.array(numpoint,dtype=np.int64)
numatoms=np.array(numatoms,dtype=np.int64)
# here the double is used to scal the potential with a high accuracy
initpot=0.0
if start_table<=1:
    pot=np.array(pot,dtype=np.float64).reshape(-1)
    initpot=np.sum(pot)/np.sum(numatoms)
    pot=pot-initpot*numatoms
# get the total number configuration for train/test
ntotpoint=0
for ipoint in numpoint:
    ntotpoint+=ipoint

#define golbal var
if numpoint[1]==0: 
    numpoint[0]=int(ntotpoint*ratio)
    numpoint[1]=ntotpoint-numpoint[0]

ntrain_vec=0
for ipoint in range(numpoint[0]):
    ntrain_vec+=numatoms[ipoint]*3

ntest_vec=0
for ipoint in range(numpoint[0],ntotpoint):
    ntest_vec+=numatoms[ipoint]*3


# parallel process the variable  
#=====================environment for select the GPU in free=================================================
local_rank = int(os.environ.get("LOCAL_RANK"))
local_size = int(os.environ.get("LOCAL_WORLD_SIZE"))

if local_size==1 and local_rank==0: gpu_sel()

world_size = int(os.environ.get("WORLD_SIZE"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu",local_rank)
dist.init_process_group(backend=DDP_backend)
a=torch.empty(100000,device=device)  # used for apply some memory to prevent two process on the smae gpu

# device the batchsize to each rank
batchsize=int(batchsize/world_size)
# devide the work on each rank
# get the shifts and atom_index of each neighbor for train
rank=dist.get_rank()
rank_begin=int(np.ceil(numpoint[0]/world_size))*rank
rank_end=min(int(np.ceil(numpoint[0]/world_size))*(rank+1),numpoint[0])
range_train=[rank_begin,rank_end]
com_coor_train,force_train,mol_train,numatoms_train,species_train,atom_index_train,shifts_train=\
get_info_of_rank(range_train,atom,atomtype,mass,mol,numatoms,scalmatrix,period_table,coor,force,\
start_table,table_coor,neigh_atoms,batchsize_train,cutoff,device,np_dtype)


## take the molecule information to calculate intermolecular interaction
atom_A = []
mass_A = []
numatoms_A = []
coor_A = []
atom_B = []
mass_B = []
numatoms_B = []
coor_B = []
atom_C = []
mass_C = []
numatoms_C = []
coor_C = []
coor_D= []
atom_D = []
mass_D = []
numatoms_D = []
coor_E = []
atom_E = []
mass_E = []
numatoms_E = []
coor_E = []
atom_E = []
mass_E = []
numatoms_E = []
coor_F = []
atom_F = []
mass_F = []
numatoms_F = []
coor_G = []
atom_G = []
mass_G = []
numatoms_G = []
for i in range(len(numatoms)):
    i_mol = mol[i]
    mol_A = i_mol[0]
    mol_B = i_mol[1]
    mol_C = i_mol[2]
    mol_D = i_mol[3]
    mol_E = i_mol[4]
    mol_F = i_mol[5]
    mol_G = i_mol[6]
    numatoms_A.append(mol_A)
    atom_A.append(atom[i][:mol_A])
    mass_A.append(mass[i][:mol_A])
    coor_A.append(coor[i][:mol_A])
    numatoms_B.append(mol_B)
    atom_B.append(atom[i][mol_A:mol_A+mol_B])
    mass_B.append(mass[i][mol_A:mol_A+mol_B])
    coor_B.append(coor[i][mol_A:mol_A+mol_B])
    if mol_C > 0:
        numatoms_C.append(mol_C)
        atom_C.append(atom[i][mol_A+mol_B:mol_A+mol_B+mol_C])
        mass_C.append(mass[i][mol_A+mol_B:mol_A+mol_B+mol_C])
        coor_C.append(coor[i][mol_A+mol_B:mol_A+mol_B+mol_C])
        if mol_D > 0:
            numatoms_D.append(mol_D)
            atom_D.append(atom[i][mol_A+mol_B+mol_C:mol_A+mol_B+mol_C+mol_D])
            mass_D.append(mass[i][mol_A+mol_B+mol_C:mol_A+mol_B+mol_C+mol_D])
            coor_D.append(coor[i][mol_A+mol_B+mol_C:mol_A+mol_B+mol_C+mol_D])
            if mol_E > 0:
                numatoms_E.append(mol_E)
                atom_E.append(atom[i][mol_A+mol_B+mol_C+mol_D:mol_A+mol_B+mol_C+mol_D+mol_E])
                mass_E.append(mass[i][mol_A+mol_B+mol_C+mol_D:mol_A+mol_B+mol_C+mol_D+mol_E])
                coor_E.append(coor[i][mol_A+mol_B+mol_C+mol_D:mol_A+mol_B+mol_C+mol_D+mol_E])
                if mol_F > 0:
                    numatoms_F.append(mol_F)
                    atom_F.append(atom[i][mol_A+mol_B+mol_C+mol_D+mol_E:mol_A+mol_B+mol_C+mol_D+mol_E+mol_F])
                    mass_F.append(mass[i][mol_A+mol_B+mol_C+mol_D+mol_E:mol_A+mol_B+mol_C+mol_D+mol_E+mol_F])
                    coor_F.append(coor[i][mol_A+mol_B+mol_C+mol_D+mol_E:mol_A+mol_B+mol_C+mol_D+mol_E+mol_F])
                    if mol_G > 0:
                        numatoms_G.append(mol_G)
                        atom_G.append(atom[i][mol_A+mol_B+mol_C+mol_D+mol_E+mol_F:mol_A+mol_B+mol_C+mol_D+mol_E+mol_F+mol_G])
                        mass_G.append(mass[i][mol_A+mol_B+mol_C+mol_D+mol_E+mol_F:mol_A+mol_B+mol_C+mol_D+mol_E+mol_F+mol_G])
                        coor_G.append(coor[i][mol_A+mol_B+mol_C+mol_D+mol_E+mol_F:mol_A+mol_B+mol_C+mol_D+mol_E+mol_F+mol_G])
                    else:
                        numatoms_G.append(mol_A)
                        atom_G.append(atom[i][:mol_A])
                        mass_G.append(mass[i][:mol_A])
                        coor_G.append(coor[i][:mol_A])                          
                else:
                    numatoms_F.append(mol_A)
                    atom_F.append(atom[i][:mol_A])
                    mass_F.append(mass[i][:mol_A])
                    coor_F.append(coor[i][:mol_A])    
                    numatoms_G.append(mol_A)
                    atom_G.append(atom[i][:mol_A])
                    mass_G.append(mass[i][:mol_A])
                    coor_G.append(coor[i][:mol_A])     
            else:
                numatoms_E.append(mol_A)
                atom_E.append(atom[i][:mol_A])
                mass_E.append(mass[i][:mol_A])
                coor_E.append(coor[i][:mol_A])
                numatoms_F.append(mol_A)
                atom_F.append(atom[i][:mol_A])
                mass_F.append(mass[i][:mol_A])
                coor_F.append(coor[i][:mol_A])
                numatoms_G.append(mol_A)
                atom_G.append(atom[i][:mol_A])
                mass_G.append(mass[i][:mol_A])
                coor_G.append(coor[i][:mol_A])       
        else:
            numatoms_D.append(mol_A)
            atom_D.append(atom[i][:mol_A])
            mass_D.append(mass[i][:mol_A])
            coor_D.append(coor[i][:mol_A])
            numatoms_E.append(mol_A)
            atom_E.append(atom[i][:mol_A])
            mass_E.append(mass[i][:mol_A])
            coor_E.append(coor[i][:mol_A])
            numatoms_F.append(mol_A)
            atom_F.append(atom[i][:mol_A])
            mass_F.append(mass[i][:mol_A])
            coor_F.append(coor[i][:mol_A])  
            numatoms_G.append(mol_A)
            atom_G.append(atom[i][:mol_A])
            mass_G.append(mass[i][:mol_A])
            coor_G.append(coor[i][:mol_A])              
    else:
        numatoms_C.append(mol_A)
        atom_C.append(atom[i][:mol_A])
        mass_C.append(mass[i][:mol_A])
        coor_C.append(coor[i][:mol_A])
        numatoms_D.append(mol_A)
        atom_D.append(atom[i][:mol_A])
        mass_D.append(mass[i][:mol_A])
        coor_D.append(coor[i][:mol_A])
        numatoms_E.append(mol_A)
        atom_E.append(atom[i][:mol_A])
        mass_E.append(mass[i][:mol_A])
        coor_E.append(coor[i][:mol_A])
        numatoms_F.append(mol_A)
        atom_F.append(atom[i][:mol_A])
        mass_F.append(mass[i][:mol_A])
        coor_F.append(coor[i][:mol_A])
        numatoms_G.append(mol_A)
        atom_G.append(atom[i][:mol_A])
        mass_G.append(mass[i][:mol_A])
        coor_G.append(coor[i][:mol_A])    
numatoms_A = np.array(numatoms_A)
numatoms_B = np.array(numatoms_B)
numatoms_C = np.array(numatoms_C)
numatoms_D = np.array(numatoms_D)
numatoms_E = np.array(numatoms_E)
numatoms_F = np.array(numatoms_F)
numatoms_G = np.array(numatoms_G)

com_coor_A_train,force_A_train,mol_A_train,numatoms_A_train,species_A_train,atom_A_index_train,shifts_A_train=\
get_info_of_rank(range_train,atom_A,atomtype,mass_A,mol,numatoms_A, scalmatrix,period_table,coor_A,force,\
start_table,table_coor,neigh_atoms,batchsize_train,cutoff,device,np_dtype)

com_coor_B_train,force_B_train,mol_B_train,numatoms_B_train,species_B_train,atom_B_index_train,shifts_B_train=\
get_info_of_rank(range_train,atom_B,atomtype,mass_B,mol,numatoms_B, scalmatrix,period_table,coor_B,force,\
start_table,table_coor,neigh_atoms,batchsize_train,cutoff,device,np_dtype)

com_coor_C_train,force_C_train,mol_C_train,numatoms_C_train,species_C_train,atom_C_index_train,shifts_C_train=\
get_info_of_rank(range_train,atom_C,atomtype,mass_C,mol,numatoms_C, scalmatrix,period_table,coor_C,force,\
start_table,table_coor,neigh_atoms,batchsize_train,cutoff,device,np_dtype)

com_coor_D_train,force_D_train,mol_D_train,numatoms_D_train,species_D_train,atom_D_index_train,shifts_D_train=\
get_info_of_rank(range_train,atom_D,atomtype,mass_D,mol,numatoms_D, scalmatrix,period_table,coor_D,force,\
start_table,table_coor,neigh_atoms,batchsize_train,cutoff,device,np_dtype)

com_coor_E_train,force_E_train,mol_E_train,numatoms_E_train,species_E_train,atom_E_index_train,shifts_E_train=\
get_info_of_rank(range_train,atom_E,atomtype,mass_E,mol,numatoms_E, scalmatrix,period_table,coor_E,force,\
start_table,table_coor,neigh_atoms,batchsize_train,cutoff,device,np_dtype)

com_coor_F_train,force_F_train,mol_F_train,numatoms_F_train,species_F_train,atom_F_index_train,shifts_F_train=\
get_info_of_rank(range_train,atom_F,atomtype,mass_F,mol,numatoms_F, scalmatrix,period_table,coor_F,force,\
start_table,table_coor,neigh_atoms,batchsize_train,cutoff,device,np_dtype)

com_coor_G_train,force_G_train,mol_G_train,numatoms_G_train,species_G_train,atom_G_index_train,shifts_G_train=\
get_info_of_rank(range_train,atom_G,atomtype,mass_G,mol,numatoms_G, scalmatrix,period_table,coor_G,force,\
start_table,table_coor,neigh_atoms,batchsize_train,cutoff,device,np_dtype)

# get the shifts and atom_index of each neighbor for test
rank_begin=int(np.ceil(numpoint[1]/world_size))*rank
rank_end=min(int(np.ceil(numpoint[1]/world_size))*(rank+1),numpoint[1])
range_test=[numpoint[0]+rank_begin,numpoint[0]+rank_end]
com_coor_test,force_test,mol_test,numatoms_test,species_test,atom_index_test,shifts_test=\
get_info_of_rank(range_test,atom,atomtype,mass,mol,numatoms,scalmatrix,period_table,coor,force,\
start_table,table_coor,neigh_atoms,batchsize_test,cutoff,device,np_dtype)

com_coor_A_test,force_A_test,mol_A_test,numatoms_A_test,species_A_test,atom_A_index_test,shifts_A_test=\
get_info_of_rank(range_test,atom_A,atomtype,mass_A,mol,numatoms_A, scalmatrix,period_table,coor_A,force,\
start_table,table_coor,neigh_atoms,batchsize_test,cutoff,device,np_dtype)

com_coor_B_test,force_B_test,mol_B_test,numatoms_B_test,species_B_test,atom_B_index_test,shifts_B_test=\
get_info_of_rank(range_test,atom_B,atomtype,mass_B,mol,numatoms_B, scalmatrix,period_table,coor_B,force,\
start_table,table_coor,neigh_atoms,batchsize_test,cutoff,device,np_dtype)

com_coor_C_test,force_C_test,mol_C_test,numatoms_C_test,species_C_test,atom_C_index_test,shifts_C_test=\
get_info_of_rank(range_test,atom_C,atomtype,mass_C,mol,numatoms_C, scalmatrix,period_table,coor_C,force,\
start_table,table_coor,neigh_atoms,batchsize_test,cutoff,device,np_dtype)

com_coor_D_test,force_D_test,mol_D_test,numatoms_D_test,species_D_test,atom_D_index_test,shifts_D_test=\
get_info_of_rank(range_test,atom_D,atomtype,mass_D,mol,numatoms_D, scalmatrix,period_table,coor_D,force,\
start_table,table_coor,neigh_atoms,batchsize_test,cutoff,device,np_dtype)

com_coor_E_test,force_E_test,mol_E_test,numatoms_E_test,species_E_test,atom_E_index_test,shifts_E_test=\
get_info_of_rank(range_test,atom_E,atomtype,mass_E,mol,numatoms_E, scalmatrix,period_table,coor_E,force,\
start_table,table_coor,neigh_atoms,batchsize_test,cutoff,device,np_dtype)

com_coor_F_test,force_F_test,mol_F_test,numatoms_F_test,species_F_test,atom_F_index_test,shifts_F_test=\
get_info_of_rank(range_test,atom_F,atomtype,mass_F,mol,numatoms_F, scalmatrix,period_table,coor_F,force,\
start_table,table_coor,neigh_atoms,batchsize_test,cutoff,device,np_dtype)

com_coor_G_test,force_G_test,mol_G_test,numatoms_G_test,species_G_test,atom_G_index_test,shifts_G_test=\
get_info_of_rank(range_test,atom_G,atomtype,mass_G,mol,numatoms_G, scalmatrix,period_table,coor_G,force,\
start_table,table_coor,neigh_atoms,batchsize_test,cutoff,device,np_dtype)
# nprop is the number of properties used for training in the same NN  if start_table==1: nprop=2 else nprop=1
nprop=1
if start_table==1: 
    pot_train=torch.from_numpy(np.array(pot[range_train[0]:range_train[1]],dtype=np_dtype))
    pot_test=torch.from_numpy(np.array(pot[range_test[0]:range_test[1]],dtype=np_dtype))
    abpropset_train=(pot_train,force_train)
    abpropset_test=(pot_test,force_test)
    nprop=2
    test_nele=torch.empty(nprop)
    train_nele=torch.empty(nprop)
    train_nele[0]=numpoint[0]
    train_nele[1]=ntrain_vec
    test_nele[0]=numpoint[1] 
    test_nele[1]=ntest_vec
   
if start_table==0: 
    pot_train=torch.from_numpy(np.array(pot[range_train[0]:range_train[1]],dtype=np_dtype))
    pot_test=torch.from_numpy(np.array(pot[range_test[0]:range_test[1]],dtype=np_dtype))
    abpropset_train=(pot_train,)
    abpropset_test=(pot_test,)
    test_nele=torch.empty(nprop)
    train_nele=torch.empty(nprop)
    train_nele[0]=numpoint[0] 
    test_nele[0]=numpoint[1] 

if start_table==2 or start_table==3: 
    dip_train=torch.from_numpy(np.array(dip[range_train[0]:range_train[1]],dtype=np_dtype))
    dip_test=torch.from_numpy(np.array(dip[range_test[0]:range_test[1]],dtype=np_dtype))
    abpropset_train=(dip_train,)
    abpropset_test=(dip_test,)
    test_nele=torch.empty(nprop)
    train_nele=torch.empty(nprop)
    train_nele[0]=numpoint[0]*3
    test_nele[0]=numpoint[1]*3

if start_table==4: 
    pol_train=torch.from_numpy(np.array(pol[range_train[0]:range_train[1]],dtype=np_dtype))
    pol_test=torch.from_numpy(np.array(pol[range_test[0]:range_test[1]],dtype=np_dtype))
    abpropset_train=(pol_train,)
    abpropset_test=(pol_test,)
    test_nele=torch.empty(nprop)
    train_nele=torch.empty(nprop)
    train_nele[0]=numpoint[0]*9
    test_nele[0]=numpoint[1]*9 

# delete the original coordiante
del coor,mass,numatoms,atom,scalmatrix,period_table
if start_table==0: del pot
if start_table==1: del pot,force
if start_table==2 and start_table==3: del dip
if start_table==4: del pol
gc.collect()
    
#======================================================
# random list of index
prop_ceff=torch.ones(2,device=device)
prop_ceff[0]=e_ceff
prop_ceff[1]=init_f
patience_epoch=patience_epoch/print_epoch

# dropout_p for each hidden layer
dropout_p=np.array(dropout_p,dtype=np_dtype)
