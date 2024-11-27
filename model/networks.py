import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init
from model_def import DDPM  # Ensure the DDPM model is imported

logger = logging.getLogger('base')


# Initialize weights for various layers
def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{:s}] not implemented'.format(init_type))


# Define the Graph and Diffusion Models (GCLAD + DDPM)
def define_G(opt):
    model_opt = opt['model']
    
    # Initialize DDPM first
    ddpm_opts = {
        'model': {'beta_schedule': {'train': model_opt['beta_schedule']['train']}},
        'train': {'optimizer': {'lr': model_opt['train']['lr']}},
        'path': {'resume_state': None},
        'phase': 'train'
    }

    ddpm = DDPM(ddpm_opts).to(opt['gpu_ids'][0] if opt['gpu_ids'] is not None else 'cpu')

    # Now initialize GCLAD model
    if model_opt['which_model_G'] == 'gclad':
        from model_def import GCLAD  # Assuming your GCLAD model is in a separate file
        
        model = GCLAD(
            n_in=model_opt['gclad']['n_in'], 
            n_h=model_opt['gclad']['n_h'],
            activation=model_opt['gclad']['activation'],
            negsamp_round_patch=model_opt['gclad']['negsamp_round_patch'],
            negsamp_round_context=model_opt['gclad']['negsamp_round_context'],
            readout=model_opt['gclad']['readout'],
            opt=ddpm_opts  # Pass DDPM model options here
        ).to(opt['gpu_ids'][0] if opt['gpu_ids'] is not None else 'cpu')

        # Set DDPM for GCLAD model
        model.set_ddpm(ddpm)
        
        # Initialize model weights
        if opt['phase'] == 'train':
            init_weights(model, init_type='kaiming')

    if opt['gpu_ids'] and opt['distributed']:
        assert torch.cuda.is_available()
        model = nn.DataParallel(model)

    return model
