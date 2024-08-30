#! /usr/bin/env python3
import time
from src.read import *
from src.dataloader import *
from src.optimize import *
from src.density import *
from src.MODEL import *
from src.EMA import *
if activate=='Tanh_like':
    from src.activate import Tanh_like as actfun
else:
    from src.activate import Relu_like as actfun

if start_table==0:
    from src.Property_energy import *
elif start_table==1:
    from src.Property_force import *
elif start_table==2:
    from src.Property_DM import *
elif start_table==3:
    from src.Property_TDM import *
elif start_table==4:
    from src.Property_POL import *
from src.cpu_gpu import *
from src.Loss import *
PES_Lammps=None
if start_table<=1:
    import pes.script_PES as PES_Normal
    import lammps.script_PES as PES_Lammps

elif start_table==2:
    import dm.script_PES as PES_Normal
elif start_table==3:
    import tdm.script_PES as PES_Normal
elif start_table==4:
    import pol.script_PES as PES_Normal

#==============================train data loader===================================
dataloader_train=DataLoader(com_coor_train,com_coor_A_train,com_coor_B_train,com_coor_C_train,com_coor_D_train,com_coor_E_train,com_coor_F_train,com_coor_G_train,\
                            numatoms_train,numatoms_A_train,numatoms_B_train,numatoms_C_train,numatoms_D_train,numatoms_E_train,numatoms_F_train,numatoms_G_train,\
                            species_train,species_A_train,species_B_train,species_C_train,species_D_train,species_E_train,species_F_train,species_G_train,\
                            atom_index_train,atom_A_index_train,atom_B_index_train,atom_C_index_train,atom_D_index_train,atom_E_index_train,atom_F_index_train,atom_G_index_train,\
                            shifts_train,shifts_A_train,shifts_B_train,shifts_C_train,shifts_D_train,shifts_E_train,shifts_F_train,shifts_G_train,\
                            abpropset_train,mol_train,batchsize_train,shuffle=True)
#=================================test data loader=================================
dataloader_test=DataLoader(com_coor_test,com_coor_A_test,com_coor_B_test,com_coor_C_test,com_coor_D_test,com_coor_E_test,com_coor_F_test,com_coor_G_test,\
                            numatoms_test,numatoms_A_test,numatoms_B_test,numatoms_C_test,numatoms_D_test,numatoms_E_test,numatoms_F_test,numatoms_G_test,\
                            species_test,species_A_test,species_B_test,species_C_test,species_D_test,species_E_test,species_F_test,species_G_test,\
                            atom_index_test,atom_A_index_test,atom_B_index_test,atom_C_index_test,atom_D_index_test,atom_E_index_test,atom_F_index_test,atom_G_index_test,\
                            shifts_test,shifts_A_test,shifts_B_test,shifts_C_test,shifts_D_test,shifts_E_test,shifts_F_test,shifts_G_test,\
                            abpropset_test,mol_test,batchsize_test,shuffle=False)

# dataloader used for load the mini-batch data
if torch.cuda.is_available(): 
    data_train=CudaDataLoader(dataloader_train,device,queue_size=queue_size)
    data_test=CudaDataLoader(dataloader_test,device,queue_size=queue_size)
else:
    data_train=dataloader_train
    data_test=dataloader_test
#=======================density======================================================
getdensity=GetDensity(rs,inta,cutoff,neigh_atoms,nipsin,norbit)
#==============================nn module=================================
nnmod=NNMod(maxnumtype,outputneuron,atomtype,nblock,list(nl),dropout_p,actfun,initpot=initpot,table_norm=table_norm)
nnmodlist=[nnmod]
if start_table == 4:
    nnmod1=NNMod(maxnumtype,outputneuron,atomtype,nblock,list(nl),dropout_p,actfun,table_norm=table_norm)
    nnmod2=NNMod(maxnumtype,outputneuron,atomtype,nblock,list(nl),dropout_p,actfun,table_norm=table_norm)
    nnmodlist.append(nnmod1)
    nnmodlist.append(nnmod2)
#=========================create the module=========================================
Prop_class=Property(getdensity,nnmodlist).to(device)  # to device must be included

##  used for syncbn to synchronizate the mean and variabce of bn 
#Prop_class=torch.nn.SyncBatchNorm.convert_sync_batchnorm(Prop_class).to(device)
if world_size>1:
    if torch.cuda.is_available():
        Prop_class = DDP(Prop_class, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=find_unused)
    else:
        Prop_class = DDP(Prop_class, find_unused_parameters=find_unused)

#define the loss function
loss_fn=Loss()

#define optimizer
optim=torch.optim.AdamW(Prop_class.parameters(), lr=start_lr, weight_decay=re_ceff)

# learning rate scheduler 
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optim,factor=decay_factor,patience=patience_epoch,min_lr=end_lr)

# load the model from EANN.pth
if table_init==1:
    if torch.cuda.is_available():
        device1=device
    else:
        device1="cpu"
    checkpoint = torch.load("EANN.pth",map_location=torch.device(device1))
    Prop_class.load_state_dict(checkpoint['eannparam'])
    optim.load_state_dict(checkpoint['optimizer'])
    if optim.param_groups[0]["lr"]>start_lr: optim.param_groups[0]["lr"]=start_lr  #for restart with a learning rate 
    if optim.param_groups[0]["lr"]<end_lr: optim.param_groups[0]["lr"]=start_lr  #for restart with a learning rate 
    lr=optim.param_groups[0]["lr"]
    f_ceff=init_f+(final_f-init_f)*(lr-start_lr)/(end_lr-start_lr+1e-8)
    prop_ceff[1]=f_ceff


ema = EMA(Prop_class, 0.999)
#==========================================================
if dist.get_rank()==0:
    fout.write(time.strftime("%Y-%m-%d-%H_%M_%S \n", time.localtime()))
    fout.flush()
    for name, m in Prop_class.named_parameters():
        print(name)
#==========================================================
Optimize(fout,prop_ceff,nprop,train_nele,test_nele,init_f,final_f,start_lr,end_lr,print_epoch,Epoch,\
data_train,data_test,Prop_class,loss_fn,optim,scheduler,ema,PES_Normal,device,PES_Lammps=PES_Lammps)
if dist.get_rank()==0:
    fout.write(time.strftime("%Y-%m-%d-%H_%M_%S \n", time.localtime()))
    fout.write("terminated normal\n")
    fout.close()
