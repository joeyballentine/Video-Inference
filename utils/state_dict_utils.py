from utils.model_classes.SOFVSR_model import SOFVSRModel, SOFVSR_RRDB_Model
from utils.model_classes.RIFE_model import RIFEModel, RIFE_HD_Model
from utils.model_classes.TecoGAN_model import TecoGanModel


def get_model_from_state_dict(state_dict, device):
    # Automatic scale detection & arch detection
    keys = state_dict.keys()

    # RIFE
    if 'flownet.block0.conv0.0.weight' in keys:
        # HD RIFE model
        if 'flownet.block3.conv0.0.weight' in keys:
            return RIFE_HD_Model(device=device) 
        # Regular RIFE model
        else:
            return RIFEModel(device=device)
    # TecoGAN
    elif 'fnet.encoder1.0.weight' in keys:
        # This isn't a guarantee, it just works with the provided models
        # TODO: Extract nb/nf from state dict
        if 'upscample_func.kernels' in keys:
            return TecoGanModel(device=device, degradation='BD')
        else:
            return TecoGanModel(device=device, degradation='BI')
    # SOFVSR
    else:
        # Extract num_channels
        num_channels = state_dict['OFR.RNN1.0.weight'].shape[0]

        # ESRGAN RRDB SR net
        if 'SR.model.1.sub.0.RDB1.conv1.0.weight' in keys:
            # extract model information
            scale2 = 0
            max_part = 0
            for part in list(state_dict):
                if part.startswith('SR.'):
                    parts = part.split('.')[1:]
                    n_parts = len(parts)
                    if n_parts == 5 and parts[2] == 'sub':
                        nb = int(parts[3])
                    elif n_parts == 3:
                        part_num = int(parts[1])
                        if part_num > 6 and parts[2] == 'weight':
                            scale2 += 1
                        if part_num > max_part:
                            max_part = part_num
                            out_nc = state_dict[part].shape[0]
            scale = 2 ** scale2
            in_nc = state_dict['SR.model.0.weight'].shape[1]
            nf = state_dict['SR.model.0.weight'].shape[0]

            if scale == 2:
                if state_dict['OFR.SR.1.weight'].shape[0] == 576:
                    scale = 3

            frame_size = state_dict['SR.model.0.weight'].shape[1]
            num_frames = (((frame_size - 3) // (3 * (scale ** 2))) + 1)

            return SOFVSR_RRDB_Model(only_y=False, scale=scale, num_frames=num_frames, num_channels=num_channels,
                                SR_net='rrdb', sr_nf=nf, sr_nb=nb, img_ch=3, sr_gaussian_noise=False, device=device)
        # Default SOFVSR SR net
        else:
            if 'OFR.SR.3.weight' in keys:
                scale = 1
            elif 'SR.body.6.bias' in keys:
                # 2 and 3 share the same architecture keys so here we check the shape
                if state_dict['SR.body.3.weight'].shape[0] == 256:
                    scale = 2
                elif state_dict['SR.body.3.weight'].shape[0] == 576:
                    scale = 3
            elif 'SR.body.9.bias' in keys:
                scale = 4
            else:
                raise ValueError('Scale could not be determined from model')
            # Extract num_frames from model
            frame_size = state_dict['SR.body.0.weight'].shape[1]
            num_frames = (((frame_size - 1) // scale ** 2) + 1)
            return SOFVSRModel(only_y=True, scale=scale, num_frames=num_frames,
                                num_channels=num_channels, SR_net='sofvsr', img_ch=1, device=device)
            