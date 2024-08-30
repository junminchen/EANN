import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist


def Optimize(fout,prop_ceff,nprop,train_nele,test_nele,init_f,final_f,start_lr,end_lr,print_epoch,Epoch,data_train,data_test,\
Prop_class,loss_fn,optim,scheduler,ema,PES_Normal,device,PES_Lammps=None): 

    rank=dist.get_rank()
    best_loss=1e30*torch.ones(1,device=device)    

    for iepoch in range(Epoch): 
        # set the model to train
       Prop_class.train()
       lossprop=torch.zeros(nprop,device=device)        
       for data in data_train:
          abProp,mol,cart,cart_A,cart_B,cart_C,cart_D,cart_E,cart_F,cart_G,\
          numatoms,numatoms_A,numatoms_B,numatoms_C,numatoms_D,numatoms_E,numatoms_F,numatoms_G,\
          species,species_A,species_B,species_C,species_D,species_E,species_F,species_G,\
          atom_index,atom_index_A,atom_index_B,atom_index_C,atom_index_D,atom_index_E,atom_index_F,atom_index_G,\
          shifts,shifts_A,shifts_B,shifts_C,shifts_D,shifts_E,shifts_F,shifts_G=data
          loss=loss_fn(Prop_class(cart,numatoms,species,atom_index,shifts),\
                        Prop_class(cart_A,numatoms_A,species_A,atom_index_A,shifts_A),\
                        Prop_class(cart_B,numatoms_B,species_B,atom_index_B,shifts_B),\
                        Prop_class(cart_C,numatoms_C,species_C,atom_index_C,shifts_C),\
                        Prop_class(cart_D,numatoms_D,species_D,atom_index_D,shifts_D),\
                        Prop_class(cart_E,numatoms_E,species_E,atom_index_E,shifts_E),\
                        Prop_class(cart_F,numatoms_F,species_F,atom_index_F,shifts_F),\
                        Prop_class(cart_G,numatoms_G,species_G,atom_index_G,shifts_G),\
                        mol,\
                        abProp)
          lossprop+=loss.detach()
          loss=torch.sum(torch.mul(loss,prop_ceff[0:nprop]))
          # clear the gradients of param
          #optim.zero_grad()
          optim.zero_grad(set_to_none=True)
          #print(torch.cuda.memory_allocated)
          # obtain the gradients
          loss.backward()
          optim.step()   

          #doing the exponential moving average update the EMA parameters
          ema.update()
    
       #  print the error of vailadation and test each print_epoch
       if np.mod(iepoch,print_epoch)==0:
          # apply the EMA parameters to evaluate
          ema.apply_shadow()
          # set the model to eval for used in the model
          Prop_class.eval()
          # all_reduce the rmse form the training process 
          # here we dont need to recalculate the training error for saving the computation
          dist.all_reduce(lossprop,op=dist.ReduceOp.SUM)
          loss=torch.sum(lossprop)
          
          # get the current rank and print the error in rank 0
          if rank==0:
              lossprop=torch.sqrt(lossprop.detach().cpu()/train_nele)
              lr=optim.param_groups[0]["lr"]
              fout.write("{:5} {:4} {:15} {:5e}  {} ".format("Epoch=",iepoch,"learning rate",lr,"train error:"))
              for error in lossprop:
                  fout.write('{:10.5f} '.format(error))
          
          # calculate the test error
          lossprop=torch.zeros(nprop,device=device)
          for data in data_test:
            abProp,mol,cart,cart_A,cart_B,cart_C,cart_D,cart_E,cart_F,cart_G,\
            numatoms,numatoms_A,numatoms_B,numatoms_C,numatoms_D,numatoms_E,numatoms_F,numatoms_G,\
            species,species_A,species_B,species_C,species_D,species_E,species_F,species_G,\
            atom_index,atom_index_A,atom_index_B,atom_index_C,atom_index_D,atom_index_E,atom_index_F,atom_index_G,\
            shifts,shifts_A,shifts_B,shifts_C,shifts_D,shifts_E,shifts_F,shifts_G=data
            loss=loss_fn(Prop_class(cart,numatoms,species,atom_index,shifts),\
                            Prop_class(cart_A,numatoms_A,species_A,atom_index_A,shifts_A),\
                            Prop_class(cart_B,numatoms_B,species_B,atom_index_B,shifts_B),\
                            Prop_class(cart_C,numatoms_C,species_C,atom_index_C,shifts_C),\
                            Prop_class(cart_D,numatoms_D,species_D,atom_index_D,shifts_D),\
                            Prop_class(cart_E,numatoms_E,species_E,atom_index_E,shifts_E),\
                            Prop_class(cart_F,numatoms_F,species_F,atom_index_F,shifts_F),\
                            Prop_class(cart_G,numatoms_G,species_G,atom_index_G,shifts_G),\
                            mol,\
                            abProp)
            lossprop=lossprop+loss.detach()
        #   for data in data_test:
        #      abProp,cart,numatoms,species,atom_index,shifts=data
        #      loss=loss_fn(Prop_class(cart,numatoms,species,atom_index,shifts,\
        #      create_graph=False),abProp)
        #      lossprop=lossprop+loss.detach()

          # all_reduce the rmse
          dist.all_reduce(lossprop,op=dist.ReduceOp.SUM)
          loss=torch.sum(lossprop)
          scheduler.step(loss)
          lr=optim.param_groups[0]["lr"]
          f_ceff=init_f+(final_f-init_f)*(lr-start_lr)/(end_lr-start_lr+1e-8)
          prop_ceff[1]=f_ceff
          #  save the best model
          if lossprop[0]<best_loss[0]:
             if rank == 0:
                 state = {'eannparam': Prop_class.state_dict(), 'optimizer': optim.state_dict()}
                 torch.save(state, "./EANN.pth")
                 best_loss[0]=lossprop[0]
                 PES_Normal.jit_pes()
                 if PES_Lammps:
                     PES_Lammps.jit_pes()
          
          # restore the model for continue training
          ema.restore()
          if rank==0:
              lossprop=torch.sqrt(lossprop.detach().cpu()/test_nele)
              fout.write('{} '.format("test error:"))
              for error in lossprop:
                 fout.write('{:10.5f} '.format(error))
              # if stop criterion
              fout.write("\n")
              fout.flush()
          if lr==end_lr: break

