#%%
import torch

## define constants for the model

# diffusion constants and advection flow rate
d1, d2, d3, flow_rate = 120.90, 1.8183, 0.015150, 7.75

# state variable removal rates
evap_rate, seep_rate, mort_rate = 0.2121, 0.1212, 0.35

# water infiltration and plant water uptake rates
infilt_rate, uptake_rate = 2.1, 1.9

# precipitation
precip = 1.845

# water to plant biomass conversion
water_efficiency = 0.449

# plant growth nonlinearity factors
eta, q = 0, 0

# effect of vegetation on temperature
alpha = 3.75

# define model class
class NCA(torch.nn.Module):
  def __init__(self, solved=True):
    super().__init__()


    # timestep size
    self.ts = 0.0001

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

        
  def forward(self, w, b, T):

    # discretized differential equations of the reaction-diffusion system
    self.dw = ((self.d2*self.D2(w) - self.seep_rate * w + self.precip - self.uptake_rate * w * b * (1 + self.eta * b**self.q)))*self.ts

    self.db = ((self.d3*self.D3(b)- self.mort_rate*T * b + self.water_efficiency * self.uptake_rate * w * b * (1 + self.eta * b**self.q)))*self.ts

    # update cell states
    
    w = w + self.dw
    b = b + self.db

    return w, b
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


# %%
# simulate the model for 100 timesteps to see if it works
n_steps = 1000000

h, w, b = h0, w0, b0
for i in range(n_steps):
   T = torch.ones(height,width)*8
   T = T - (alpha*b)
   w, b = true_model.forward(w, b, T)
   
   if i%1000==0:
    # Create subplots: 1 row, 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Adjust figsize as needed

    # First plot
    axes[0].imshow(b.detach().cpu().squeeze())
    axes[0].set_title("Plot 1")  # Optional title
    axes[0].colorbar = fig.colorbar(axes[0].imshow(b.detach().cpu().squeeze(), vmin = 0, vmax=0.4), ax=axes[0])

    # Second plot
    im = axes[1].imshow(T.detach().cpu().squeeze(), cmap='inferno')
    axes[1].set_title("Plot 2")  # Optional title
    fig.colorbar(im, ax=axes[1])

    # Show the plots
    plt.tight_layout()
    plt.show()



# %%
