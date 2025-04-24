import torch
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F

'''siamrpn++'''
from siamRPNPP import SiamRPNPP
from data_utils import normalize
from torch.cuda.amp import autocast, GradScaler

'''hyper-parameters, which may need to be tuned'''
cls_thres = 0.7
scaler = GradScaler()


class GsearchL2500regressModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        '''make change here'''
        parser.set_defaults(norm='batch', netG='unet_128', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='lsgan')
            parser.add_argument('--lambda_L1', type=float, default=500, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L2', 'gradcam'] # 'kl1', 'kl2',
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.visual_names = ['search_clean_vis', 'search_adv_vis']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(3, 3, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionL2 = torch.nn.MSELoss()
            self.init_weight_L2 = opt.lambda_L1
            self.init_weight_cls = 0.1
            self.init_weight_reg = 1
            self.cls_margin = -5
            self.side_margin1 = -5
            self.side_margin2 = -5
            self.weight_L2 = self.init_weight_L2
            self.weight_cls = self.init_weight_cls
            self.weight_reg = self.init_weight_reg
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
        '''siamrpn++'''
        self.siam = SiamRPNPP()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.template = input[0].squeeze(0).cuda()  # pytorch tensor, shape=(1,3,127,127)

        self.search_clean255 = input[1].squeeze(0).cuda()
        # pytorch tensor, shape=(N,3,255,255) [0,255]
        self.search_clean1 = normalize(self.search_clean255)
        self.num_search = self.search_clean1.size(0)
        # print('clean image shape:',self.init_frame_clean.size())

    def forward(self, target_sz=(255, 255), is_train=False):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        '''resize to (512,512)'''
        if is_train:
            search512_clean1 = torch.nn.functional.interpolate(self.search_clean1, size=(256, 256), mode='bilinear')
            search512_clean2 = torch.nn.functional.interpolate(self.search_clean1, size=(320, 320), mode='bilinear')
            search512_adv1 = search512_clean1 + self.netG(search512_clean1)
            search512_adv2 = search512_clean2 + self.netG(search512_clean2)
            '''Then crop back to (255,255)'''
            self.search_adv1 = torch.nn.functional.interpolate(search512_adv1, size=target_sz, mode='bilinear')
            self.search_adv2 = torch.nn.functional.interpolate(search512_adv2, size=target_sz, mode='bilinear')
            self.search_adv255 = (self.search_adv1 + self.search_adv2) / 2 * 127.5 + 127.5
        else:
            search512_clean1 = torch.nn.functional.interpolate(self.search_clean1, size=(512, 512), mode='bilinear')
            search512_adv1 = search512_clean1 + self.netG(search512_clean1)
            self.search_adv1 = torch.nn.functional.interpolate(search512_adv1, size=target_sz, mode='bilinear')
            self.search_adv255 = self.search_adv1 * 127.5 + 127.5
        '''for visualization'''
        self.search_clean_vis = self.search_clean1[0:1]
        self.search_adv_vis = self.search_adv1[0:1]

    def transform(self, patch_clean1, target_sz=(255, 255)):
        '''resize to (512,512)'''
        patch512_clean1 = torch.nn.functional.interpolate(patch_clean1, size=(256, 256), mode='bilinear')
        patch512_adv1 = patch512_clean1 + self.netG(patch512_clean1)  # Residual form: G(A)+A
        patch_adv1 = torch.nn.functional.interpolate(patch512_adv1, size=target_sz, mode='bilinear')
        patch_adv255 = patch_adv1 * 127.5 + 127.5
        return patch_adv255

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # Second, G(A) = B
        self.loss_G_L2 = self.criterionL2(self.search_adv1, self.search_clean1) * self.weight_L2 * 1
        criterion = torch.nn.KLDivLoss(reduction='batchmean')
        # self.loss_kl1 = 10 - criterion(self.score_maps_adv.log_softmax(dim=-1),
        #                                self.score_maps_cl.softmax(dim=-1)) * 10
        # self.loss_kl2 = 10 - criterion(self.reg_res_adv.log_softmax(dim=-1), self.reg_res_cl.softmax(dim=-1)) * 100
        # self.loss_feat = 10 - criterion(self.feat_adv.log_softmax(dim=-1), self.feat_cl.softmax(dim=-1)) * 10
        self.loss_gradcam = 10 - criterion(self.cam_cl.log_softmax(dim=-1), self.cam_adv.softmax(dim=-1))*0.1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_L2 + self.loss_gradcam  # + self.loss_cls + self.loss_reg   self.loss_kl1 + self.loss_kl2 + self.loss_feat+ self.loss_kl1 + self.loss_kl2
        self.loss_G.backward()

    def optimize_parameters(self):
        '''One forward & backward pass. One update of parameters'''
        '''forward pass'''
        # 1. predict with clean template
        with torch.no_grad():
            self.siam.model.template(self.template)
        # 2. adversarial attack with GAN
        self.forward()  # compute fake image
        # 3. predict with adversarial template
        self.cam_cl, self.score_maps_cl, self.reg_res_cl = self.siam.compute_gradcam_weights_tracking(self.siam.model,self.siam.model.backbone.layer2,self.search_clean255)
        self.cam_adv, self.score_maps_adv, self.reg_res_adv = self.siam.compute_gradcam_weights_tracking(self.siam.model,self.siam.model.backbone.layer2,self.search_adv255)
        # self.feat_cl, self.score_maps_cl, self.reg_res_cl = self.siam.get_cls_reg(self.search_clean255,
        #                                                                                          False,0)
        # self.feat_adv, self.score_maps_adv, self.reg_res_adv = self.siam.get_cls_reg(self.search_adv255, False,1)
        '''backward pass'''
        # update G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights
