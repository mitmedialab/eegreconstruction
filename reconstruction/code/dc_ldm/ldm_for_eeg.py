import numpy as np
import wandb
import torch
from dc_ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import torch.nn as nn
import os
from dc_ldm.models.diffusion.plms import PLMSSampler
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch.nn.functional as F

#Modifications
from dc_ldm.models.EEGNet_Embedding_version import EEGNet_Embedding
import yaml

class eLDM:

    def __init__(
            self, 
            EEGNet_config_path,
            EEGNet_ckpt_path,
            device=torch.device('cpu'),
            pretrain_root='../pretrains/ldm/label2img',
            logger=None, 
            ddim_steps=250, 
            global_pool=True, 
            use_time_cond=True
            ):
        self.ckp_path = os.path.join(pretrain_root, 'model.ckpt') #path to pretrained LDM
        self.config_path = os.path.join(pretrain_root, 'config.yaml') #path to pretrained LDM config
        config = OmegaConf.load(self.config_path)
        config.model.params.unet_config.params.use_time_cond = use_time_cond
        config.model.params.unet_config.params.global_pool = global_pool

        self.cond_dim = config.model.params.unet_config.params.context_dim #512

        model = instantiate_from_config(config.model)
        pl_sd = torch.load(self.ckp_path, map_location="cpu")['state_dict']
       
        m, u = model.load_state_dict(pl_sd, strict=False)
        model.cond_stage_trainable = True
        #3. Exchange cond stage model with our encoder
        model.cond_stage_model = cond_stage_model(EEGNet_config_path, EEGNet_ckpt_path) #Use EEGNet+ as condition stage model

        model.ddim_steps = ddim_steps
        model.re_init_ema()
        if logger is not None:
            logger.watch(model, log="all", log_graph=False)

        model.p_channels = config.model.params.channels
        model.p_image_size = config.model.params.image_size
        model.ch_mult = config.model.params.first_stage_config.params.ddconfig.ch_mult

        self.device = device    
        self.model = model
        self.ldm_config = config
        self.pretrain_root = pretrain_root

    def finetune(self, trainers, dataset, test_dataset, bs1, lr1,
                output_path, config=None):
        config.trainer = None
        config.logger = None
        self.model.main_config = config
        self.model.output_path = output_path
        # self.model.train_dataset = dataset
        self.model.run_full_validation_threshold = 0.15
        # stage one: train the cond encoder with the pretrained one
      
        # stage one: only optimize conditional encoders
        print('\n##### Stage One: only optimize conditional encoders #####')
        dataloader = DataLoader(dataset, batch_size=bs1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        self.model.unfreeze_whole_model()
        self.model.freeze_first_stage()

        self.model.learning_rate = lr1
        self.model.train_cond_stage_only = True
        self.model.eval_avg = config.eval_avg
        trainers.fit(self.model, dataloader, val_dataloaders=test_loader)

        self.model.unfreeze_whole_model()
        
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'config': config,
                'state': torch.random.get_rng_state()

            },
            os.path.join(output_path, 'checkpoint.pth')
        )
        

    @torch.no_grad()
    def generate(self, eeg_embedding, num_samples, ddim_steps, HW=None, limit=None, state=None):
        all_samples = []
        if HW is None:
            shape = (self.ldm_config.model.params.channels, 
                self.ldm_config.model.params.image_size, self.ldm_config.model.params.image_size)
        else:
            num_resolutions = len(self.ldm_config.model.params.first_stage_config.params.ddconfig.ch_mult)
            shape = (self.ldm_config.model.params.channels,
                HW[0] // 2**(num_resolutions-1), HW[1] // 2**(num_resolutions-1))

        model = self.model.to(self.device)
        sampler = PLMSSampler(model)
        # sampler = DDIMSampler(model)
        if state is not None:
            torch.cuda.set_rng_state(state)
            
        with model.ema_scope():
            model.eval()
            for count, item in enumerate(eeg_embedding):
                if limit is not None:
                    if count >= limit:
                        break
                latent = item['eeg']
                gt_image = rearrange(item['image'], 'h w c -> 1 c h w') # h w c
                print(f"rendering {num_samples} examples in {ddim_steps} steps.")
                
                c = model.get_learned_conditioning(repeat(latent, 'h w -> c h w', c=num_samples).to(self.device))
                samples_ddim, _ = sampler.sample(S=ddim_steps, 
                                                conditioning=c,
                                                batch_size=num_samples,
                                                shape=shape,
                                                verbose=False)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                gt_image = torch.clamp((gt_image+1.0)/2.0, min=0.0, max=1.0)
                
                all_samples.append(torch.cat([gt_image, x_samples_ddim.detach().cpu()], dim=0)) # put groundtruth at first
                
        
        # display as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=num_samples+1)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        model = model.to('cpu')
        
        return grid, (255. * torch.stack(all_samples, 0).cpu().numpy()).astype(np.uint8)


#Modifications
def read_model_config(config_path: str):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        config = config["parameters"]["model"]["parameters"]
        model_config = {key: config[key]["value"] for key in config}
    return model_config

#Update forward to discard classifier head
def updated_forward(self, x):
    x = self.partial_forward(x) # [1 16 1 15] #bs x F2 x 1 x 15 (128*15)
    x = x.flatten(start_dim=1) # bs x F2*15
    x = self.embedding(x) # bs x 512
    return x

class cond_stage_model(nn.Module):
    def __init__(self, EEGNet_config_path, EEGNet_ckpt_path):
        super().__init__()
        # prepare pretrained EEGNet
        model_config = read_model_config(config_path = EEGNet_config_path) #"C:/Users/s_gue/Desktop/master_project/sven-thesis/pytorch/EEG_encoder_setup/P001_model_config.yaml")
        model = EEGNet_Embedding(**model_config) #This is the EEGNet model mapping from EEG input to embedding
        checkpoint = torch.load(EEGNet_ckpt_path)
        model.load_state_dict(checkpoint['state_dict'])
        model.forward = updated_forward.__get__(model)
        self.eegnet = model

        #Explanation: The output of the EEGNet is batch_size x 1 x 512. We need to map to the expected context size of the LDM
        #Therefore we use a 1x1 convolution that maps from depth 1 to 77
        #Note: As the EEGNet outputs a 512 dimensional vector, we avoid an extra linear mapping from 1024 to 512 (which is 0,5 million params!)
        self.channel_mapper = nn.Sequential(
            nn.Conv1d(1, 77, 1, bias=True),
        )

    def forward(self, x):
        latent_crossattn = self.eegnet(x)
        return self.channel_mapper(latent_crossattn.unsqueeze(1))
