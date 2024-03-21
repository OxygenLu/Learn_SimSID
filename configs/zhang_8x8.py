import sys

sys.path.insert(0, '..')

from dataloader.dataloader_zhang import *
from configs.base import BaseConfig

class MemoryMatrixBlockConfig():
    memory_layer_type = 'default'
    num_memory = 4    # square of num_patches
    num_slots = 200
    slot_dim = 256
    shrink_thres = 5
    mask_ratio = 0.95

class InpaintBlockConfig():
    use_memory_queue = False
    # if use memory queue
    num_slots = 200
    memory_channel = 128 * 2 * 2
    shrink_thres = 5
    drop = 0.
    mask_ratio = 0.95
    # if use memory matrix
    memory_config = MemoryMatrixBlockConfig()
    memory_config.memory_layer_type = 'default'
    memory_config.num_memory = 1
    memory_config.num_slots = num_slots
    memory_config.shrink_thres = 5
    memory_config.mask_ratio = 0.95

class Config(BaseConfig):

    memory_config = MemoryMatrixBlockConfig()
    inpaint_config = InpaintBlockConfig()

    def __init__(self):
        super(Config, self).__init__()

        #---------------------
        # Training Parameters
        #---------------------
        self.print_freq = 10
        self.device = 'cuda:0'
        self.epochs = 1000
        self.lr = 1e-4 # learning rate
        self.batch_size = 16
        self.test_batch_size = 2
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
        self.scheduler_args = dict(T_max=300, eta_min=self.lr*0.1)

        # GAN
        self.gan_lr = 1e-4
        self.discriminator_type = 'basic'
        self.enbale_gan = 0 #100
        self.lambda_gp = 10.
        self.size = 4
        self.num_layers = 5
        self.n_critic = 2
        self.sample_interval = 1000
        self.scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR
        self.scheduler_args_d = dict(T_max=200, eta_min=self.lr*0.2)

        # model
        self.img_size = 256
        self.num_patch = 8
        self.level = 4
        # self.shrink_thres = 5
        # self.initial_combine = 2 # from top to bottom
        # self.drop = 0.
        self.dist = True
        # self.num_slots = 20
        # self.mem_num_slots = 200
        # self.memory_channel = 512
        self.ops = ['concat', 'concat', 'none', 'none']

        # loss weight
        self.t_w = 0.01
        self.recon_w = 10.
        self.dist_w = 0.001
        self.g_w = 0.005
        self.d_w = 0.005

        self.data_root = '/mnt/data0/yixiao/zhanglab-chest-xrays/resized256'
        self.train_dataset = Zhang(self.data_root+'/CellData/chest_xray/train', train=True, img_size=(self.img_size, self.img_size))
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, drop_last=False)

        self.val_dataset = Zhang(self.data_root+'/CellData/chest_xray/val', train=False, img_size=(self.img_size, self.img_size), full=True)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

        self.test_dataset = Zhang(self.data_root+'/CellData/chest_xray/test', train=False, img_size=(self.img_size, self.img_size), full=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
