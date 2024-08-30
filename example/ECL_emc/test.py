# This is an example script to show how to obtain the energy and force by invoking the potential saved by the training .
# Typically, you can read the structure,mass, lattice parameters(cell) and give the correct periodic boundary condition (pbc) 
# and the index of each atom. All the information are required to store in the tensor of torch. 
# Then, you just pass these information to the calss "pes" that will output the energy and force.

import numpy as np
import torch
import sys
# used for select a unoccupied GPU
# gpu/cpu
#device = torch.device('cpu')
device = torch.device('cuda')

# same as the atomtype in the file input_density
atomtype=['H', 'Li', 'B', 'C', 'N', 'O', 'F', 'Na', 'P', 'S']
#load the serilizable model
pes=torch.jit.load('EANN_PES_DOUBLE.pt')
# FLOAT: torch.float32; DOUBLE:torch.double for using float/double in inference
pes.to(device).to(torch.double)
# set the eval mode
pes.eval()
pes=torch.jit.optimize_for_inference(pes)
# save the lattic parameters
cell=np.zeros((3,3),dtype=np.float64)
period_table=torch.tensor([0,0,0],dtype=torch.double,device=device)   # same as the pbc in the periodic boundary condition
npoint=0
rmse=torch.zeros(2,dtype=torch.double,device=device)
ene_ab = []
ene_pred = []
ene_ff = []
ene_sr = []
with open( "data_mol/test/configuration",'r') as f1:
    while True:
        string=f1.readline()
        print(string)
        if not string: break
        string=f1.readline()
        cell[0]=np.array(list(map(float,string.split())))
        string=f1.readline()
        cell[1]=np.array(list(map(float,string.split())))
        string=f1.readline()
        cell[2]=np.array(list(map(float,string.split())))
        string=f1.readline()
        species=[]
        cart=[]
        abforce=[]
        mass=[]
        string=f1.readline()
        mol = list(map(int,string.split()[1:8]))
        while True:
            string=f1.readline()
            if "abprop" in string: break
            tmp=string.split()
            tmp1=list(map(float,tmp[2:8]))
            cart.append(tmp1[0:3])
            abforce.append(tmp1[3:6])
            mass.append(float(tmp[1]))
            species.append(atomtype.index(tmp[0]))
        #sr = float(string.split()[2])
        #sr=torch.from_numpy(np.array([sr])).to(device)
        abene=float(string.split()[1])
        abene=torch.from_numpy(np.array([abene])).to(device)
        species=torch.from_numpy(np.array(species)).to(device)  # from numpy array to torch tensor
        cart=torch.from_numpy(np.array(cart)).to(device).to(torch.double)  # also float32/double
        mass=torch.from_numpy(np.array(mass)).to(device).to(torch.double)  # also float32/double
        abforce=torch.from_numpy(np.array(abforce)).to(device).to(torch.double)  # also float32/double
        tcell=torch.from_numpy(cell).to(device).to(torch.double)  # also float32/double
        molA=mol[0]
        molB=mol[1]
        energy,force=pes(period_table,cart,tcell,species,mass)
        energyA,forceA=pes(period_table,cart[:molA],tcell,species[:molA],mass[:molA])
        energyB,forceB=pes(period_table,cart[molA:molA+molB],tcell,species[molA:molA+molB],mass[molA:molA+molB])
        energy=energy.detach()-energyA.detach()-energyB.detach()
        force=force.detach()-torch.cat((forceA.detach(),forceB.detach()),dim=0)
        ene_ab.append(float(abene))
        ene_pred.append(float(energy))

ene_ab = np.array(ene_ab) #+ np.array(ene_sr)
ene_pred = np.array(ene_pred) #+ np.array(ene_sr)
#ene_sr = np.array(ene_sr) 
rmsd = np.sqrt(np.average((ene_ab - ene_pred)**2))
print('#', rmsd)
# print test data
with open('test_res.xvg', 'w') as f:
    print('#', rmsd, file=f)
    for i in range(len(ene_ab)):
        e1 = ene_pred[i]
        e0 = ene_ab[i]
        print(e0, e1, file=f)
