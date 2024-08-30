import torch 
import torch.nn as nn

class Loss(nn.Module):
   def __init__(self):
      super(Loss, self).__init__()
      # self.loss_fn=nn.MSELoss(reduction="mean")
      self.loss_fn=nn.MSELoss(reduction="sum")

   def forward(self,var,var_A,var_B,var_C,var_D,var_E,var_F,var_G,mol,ab):
      return  torch.cat([self.loss_fn(ivar-ivar_A*imol[0]-ivar_B*imol[1]-ivar_C*imol[2]-ivar_D*imol[3]-ivar_E*imol[4]-ivar_F*imol[5]-ivar_G*imol[6],iab).view(-1) 
                           for ivar,ivar_A,ivar_B,ivar_C,ivar_D,ivar_E,ivar_F,ivar_G,imol,iab in zip(var,var_A,var_B,var_C,var_D,var_E,var_F,var_G,mol,ab)])