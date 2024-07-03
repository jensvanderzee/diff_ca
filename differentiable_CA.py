#%%
import torch

## define constants for the model

# diffusion constants and advection flow rate
d1, d2, d3, flow_rate = 1.90, 0.0183, 0.065150, 7.75

# state variable removal rates
evap_rate, seep_rate, mort_rate = 0.2121, 0.1212, 0.6

# water infiltration and plant water uptake rates
infilt_rate, uptake_rate = 2.1, 1.9

# precipitation
precip = 0.545

# water to plant biomass conversion
water_efficiency = 0.449

# plant growth nonlinearity factors
eta, q = 1, 1

# define model class
class NCA(torch.nn.Module):
  def __init__(self, solved=True):
    super().__init__()


    # timestep size
    self.ts = 0.1

    # define diffusion operators of state variables as 2D convolutions
    self.h_lapl = torch.tensor([[0, 1, 0],
                                [1, -4, 1],
                                [0, 1, 0.0]])
    self.D1 = torch.nn.Conv2d(1, 1, (3,3), padding=1, padding_mode="replicate", bias=False)
    self.D1.weight = torch.nn.Parameter(self.h_lapl[None, None, :], requires_grad=False)

    self.w_lapl = torch.tensor([[0, 1, 0],
                                [1, -4, 1],
                                [0, 1, 0.0]])
    
    self.D2 = torch.nn.Conv2d(1, 1, (3,3), padding=1, padding_mode="replicate", bias=False)
    self.D2.weight = torch.nn.Parameter(self.w_lapl[None, None, :], requires_grad=False)


    self.b_lapl = torch.tensor([[0, 1, 0],
                                [1, -4, 1],
                                [0, 1, 0.0]])
    self.D3 = torch.nn.Conv2d(1, 1, (3,3), padding=1, padding_mode="replicate", bias=False)
    self.D3.weight = torch.nn.Parameter(self.b_lapl[None, None, :], requires_grad=False)


    # tools to simulate advective flow of surface water (h) along the gradients of a digital terrain model
    self.grad_x = torch.tensor([[0, 0, 0],
                                [-1, 0, 1],
                                [0, 0, 0.0]], dtype=torch.float, requires_grad=False)
    
    self.grad_y = torch.tensor([[0, 1, 0],
                                [0, 0, 0],
                                [0, -1, 0.0]], dtype=torch.float, requires_grad=False)
    self.grad_xy = torch.stack((self.grad_x, self.grad_y))

    self.gradient_x = torch.nn.Conv2d(1,1,(3,3), stride=1, padding=1, padding_mode="reflect", bias=False)
    self.gradient_x.weight = torch.nn.Parameter(self.grad_x[None, None, :], requires_grad=False)

    self.gradient_y = torch.nn.Conv2d(1,1,(3,3), stride=1, padding=1, padding_mode="reflect", bias=False)
    self.gradient_y.weight = torch.nn.Parameter(self.grad_y[None, None, :], requires_grad=False)


    # variable 'solved' indicates whether the model should be initialized using fixed constants for running in forward mode 
    # or using learnable parameters to be inferred from time series data
    if solved == False:
        self.precip = torch.rand(1).requires_grad_().float()
        self.precip = torch.nn.Parameter(self.precip)
        
        self.infilt_rate = torch.rand(1).requires_grad_().float()
        self.infilt_rate = torch.nn.Parameter(self.infilt_rate)
        
        self.flow_rate = torch.rand(1).requires_grad_().float()
        self.flow_rate = torch.nn.Parameter(self.flow_rate)
        
        self.seep_rate = torch.rand(1).requires_grad_().float()
        self.seep_rate = torch.nn.Parameter(self.seep_rate)
        
        
        self.uptake_rate = torch.rand(1).requires_grad_().float()
        self.uptake_rate = torch.nn.Parameter(self.uptake_rate)

        self.eta = torch.rand(1).requires_grad_().float()
        self.eta = torch.nn.Parameter(self.eta)
        
        
        self.mort_rate = torch.rand(1).requires_grad_().float()
        self.mort_rate = torch.nn.Parameter(self.mort_rate)
        
        self.evap_rate = torch.rand(1).requires_grad_().float()
        self.evap_rate = torch.nn.Parameter(self.evap_rate)
        
        self.water_efficiency = torch.rand(1).requires_grad_().float()
        self.water_efficiency = torch.nn.Parameter(self.water_efficiency)
        
        self.d1 = torch.rand(1).requires_grad_().float()
        self.d1 = torch.nn.Parameter(self.d1)
        
        self.d2 = torch.rand(1).requires_grad_().float()
        self.d2 = torch.nn.Parameter(self.d2)
        
        self.d3 = torch.rand(1).requires_grad_().float()
        self.d3 = torch.nn.Parameter(self.d3)
        
        self.q = torch.rand(1).requires_grad_().float()
        self.q = torch.nn.Parameter(self.q)
        
        
        
    if solved:
        self.precip = precip
    
        self.infilt_rate = infilt_rate
        self.flow_rate = flow_rate
        self.evap_rate = evap_rate
        self.seep_rate = seep_rate
        self.uptake_rate = uptake_rate
        self.eta = eta
        self.mort_rate = mort_rate
        self.water_efficiency = water_efficiency
        self.q = q
        
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3

        
  def forward(self, h, w, b, dem_x, dem_y):

    # surface water advection in x and y directions
    self.hgrad_x = self.gradient_x(h)
    self.hgrad_y = self.gradient_y(h)
    

    # discretized differential equations of the reaction-diffusion system
    self.dh = ((self.d1*self.D1(h) - self.evap_rate * h + self.precip - self.infilt_rate * h * b - self.flow_rate*((self.hgrad_x*dem_x)+(self.hgrad_y*dem_y))))*self.ts    
    
    self.dw = ((self.d2*self.D2(w) - self.seep_rate * w + self.infilt_rate * h * b - self.uptake_rate * w * b * (1 + self.eta * b**self.q)))*self.ts

    self.db = ((self.d3*self.D3(b)- self.mort_rate * b + self.water_efficiency * self.uptake_rate * w * b * (1 + self.eta * b**self.q)))*self.ts

    # update cell states
    h = h + self.dh
    w = w + self.dw
    b = b + self.db

    
    return h, w, b
# %%


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import matplotlib.pyplot as plt
height, width = 100, 100

h0, w0, b0 = torch.rand(height, width)[None, None, :], torch.rand(height, width)[None, None, :], torch.rand(height, width)[None, None, :]
h0, w0, b0 = h0.to(device), w0.to(device), b0.to(device)
plt.imshow(b0.detach().cpu().squeeze())
plt.colorbar()


# %%

# create a model instance 
true_model = NCA(solved = True).to(device)

# create a flat digital elevation model (this means no effect of topography on advection flow)
flat_dem = torch.zeros(height, width)[None, None, :].to(device)
flat_dem_x, flat_dem_y = true_model.gradient_x(flat_dem), true_model.gradient_y(flat_dem)

# %%
# simulate the model for 100 timesteps to see if it works
n_steps = 100

h, w, b = h0, w0, b0
for i in range(n_steps):
   h, w, b = true_model.forward(h, w, b, flat_dem_x, flat_dem_y)
   
plt.imshow(b.detach().cpu().squeeze())
plt.colorbar()




# %%

# sets up training loop to see if original parameter values can be retrieved from snapshots of the simulation

from tqdm import tqdm
# create a model instance with randomly initialized parameters
model = NCA(solved=False).to(device)

# number of iterations to train the model
n_iter = 800
# define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)
loss_function = torch.nn.MSELoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.999)

# initialize simulation and prediction states
sim_h, sim_w, sim_b = h0, w0, b0
pred_h, pred_w, pred_b = h0, w0, b0

# decrease this to generate more frequent data or increase for less frequent data
sample_rate = 30

#%%
# start training loop
pbar = tqdm(total=n_iter)
for i in range(n_iter):
  loss=0
  sim_h, sim_w, sim_b = h0, w0, b0
  pred_h, pred_w, pred_b = h0, w0, b0
  for k in range(n_steps):
    sim_h, sim_w, sim_b = true_model.forward(sim_h, sim_w, sim_b, flat_dem_x, flat_dem_y)
    pred_h, pred_w, pred_b = model.forward(pred_h, pred_w, pred_b, flat_dem_x, flat_dem_y)

    if k%sample_rate == 0 and k > 0:
      # compare the simulated and predicted vegetation states (b) of the true model and the trained model
      # one can add the loss of surface water (h) and soil water (w) here to improve training, but we consider these are unobservable in reality
      loss_b = loss_function(pred_b,sim_b)
      loss = loss + loss_b
  # backpropagate and update model parameters
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()   
  scheduler.step()

  # prevent model parameters from going negative
  for params in model.parameters():
                if params.data.shape == torch.Size([1]):
                    params.data = params.data.clamp(0,10000)
  pbar.update()
  pbar.set_description("Training loss: %.10f" % (loss) )
# %%
# print the true and learned parameter values
# try different parameters by changing 'd1' to 'd3' or 'infilt_rate'
print("true value:", true_model.infilt_rate, "\n", "learned value: ", model.infilt_rate.item())

# %%
