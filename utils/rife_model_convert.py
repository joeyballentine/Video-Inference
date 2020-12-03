from utils.architectures.RIFE_arch import RIFE
from utils.architectures.RIFE_HD_arch import RIFE as RIFEHD
import torch
from collections import OrderedDict

# To use this, move to root directory and place .pkl files in root directory as well
# Change mode to HD if using HD rife model
# Then just run this script
# TODO: Make better script

mode = 'HD' # options: reg, hd

if mode == 'HD':
    model = RIFEHD()
else:
    model = RIFE()

context_net = torch.load('./contextnet.pkl')
fixed_context_net = OrderedDict()
for key in context_net.keys():
    fixed_context_net[key.replace('module.', '')] = context_net[key]

flow_net = torch.load('./flownet.pkl')
fixed_flow_net = OrderedDict()
for key in flow_net.keys():
    fixed_flow_net[key.replace('module.', '')] = flow_net[key]

u_net = torch.load('./unet.pkl')
fixed_u_net = OrderedDict()
for key in u_net.keys():
    fixed_u_net[key.replace('module.', '')] = u_net[key]

model.contextnet.load_state_dict(fixed_context_net)
model.flownet.load_state_dict(fixed_flow_net)
model.fusionnet.load_state_dict(fixed_u_net)

torch.save(model.state_dict(), './rife_converted.pth', _use_new_zipfile_serialization=False)