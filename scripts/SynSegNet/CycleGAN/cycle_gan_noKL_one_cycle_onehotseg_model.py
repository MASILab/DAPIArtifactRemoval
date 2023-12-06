import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
import numpy as np
import cv2


class CycleGANNoKLOneCycleOneHotSegModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_C', type=float, default=1.0,  help='weight for segmentation loss')
            #SHUNXING
            # parser.add_argument('--lambda_A1', type=float, default=10.0, help='weight for cycle loss (A1 -> B -> A1)')
            # parser.add_argument('--lambda_B1', type=float, default=10.0, help='weight for cycle loss (B -> A1 -> B)')

            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        #self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # SHUNXING - we need to make identity A and identity A1 same. 
        # self.loss_names = ['D_A', 'G_A', 'cycle_A', 'D_B', 'G_B',  'cycle_B','marker_reproduce_B','D_A1', 'G_A1', 'cycle_A1', 'D_B1', 'G_B1', 'cycle_B1','encode_latent','avg_fake_B', 'avg_fake_B1', 'seg_B']
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'D_B', 'G_B',  'cycle_B','seg_B', 'seg_A']#,'D_A1', 'G_A1', 'cycle_A1', 'D_B1', 'G_B1', 'cycle_B1','encode_latent','avg_fake_B', 'avg_fake_B1', 'seg_B']

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        # SHUNXING
        # visual_names_A1 = ['real_A1', 'fake_B1', 'rec_A1']
        # visual_names_B1 = ['real_B1', 'fake_A1', 'rec_B1']

        # visual_names_avg_fakeB = ['avg_fakeB','Atruth','seg_B']
        visual_names_segB = ['Atruth','seg_B','Btruth','seg_A']

        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            print('do nothing')
        #    visual_names_A.append('marker_reproduce_B')
            #SHUNXING
            #visual_names_B.append('idt_A')
        #    visual_names_A1.append('marker_reproduce_B')

        # self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        #SHUNXING
        # self.visual_names = visual_names_A + visual_names_B + visual_names_A1 + visual_names_B1 + visual_names_avg_fakeB# combine visualizations for A and B
        self.visual_names = visual_names_A + visual_names_B +  visual_names_segB# combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        # if self.isTrain:
        #     self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        # else:  # during test time, only load Gs
        #     self.model_names = ['G_A', 'G_B']
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B','S_B']#, 'G_A1', 'G_B1', 'D_A1', 'D_B1']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']#, 'G_A1', 'G_B1']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # SHUNXING
        # self.netG_A1 = networks.define_G(opt.input_nc1, opt.output_nc, opt.ngf, opt.netG, opt.norm,
        #                                 not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # self.netG_B1 = networks.define_G(opt.output_nc, opt.input_nc1, opt.ngf, opt.netG, opt.norm,
        #                                 not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # self.netS_B = networks.define_G(int(opt.input_nc_seg), int(opt.output_nc_seg), opt.ngf, opt.netG, opt.norm,
        #                                 not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        print('#############')
        print(opt.input_nc)
        print(opt.output_nc)
        print('#############')

        # output channel is 3: back = 0; dapi = 1; muc2 = 2
        self.netS_B = networks.define_G(1, 2, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # self.netS_A = networks.define_G(1, 2, opt.ngf, opt.netG, opt.norm,
        #                                 not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # self.netS_B = networks.define_G(3, 1, opt.ngf, 'unet_256', opt.norm,
        #                                 not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            #SHUNXING
            # self.netD_A1 = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
            #                                 opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # self.netD_B1 = networks.define_D(opt.input_nc1, opt.ndf, opt.netD,
            #                                 opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
                    

        if self.isTrain:
        #    if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
        #        assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            #SHUNXING
            # self.fake_A1_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # self.fake_B1_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            # No need Idt because the channel number might be different
            # self.criterionIdt = torch.nn.L1Loss()

            # SHUNXING - loss for checking if different marker generates similar structure
            # self.criterionMarker = torch.nn.L1Loss()

            # SHUNXING - loss for checking the tensor latent space KL divergence
            # self.criterionEncodeLatentSpace = torch.nn.MSELoss()#CrossEntropyLoss()#KLDivLoss(reduction = 'batchmean')
            # self.criterionEncodeLatentSpace = torch.nn.KLDivLoss()

            # SHUNXING - for segmentaiton
            # self.criterionSegB = torch.nn.CrossEntropyLoss().to(self.device)

            # self.criterionSegB = torch.nn.BCELoss()

            # self.opt.cross_entropy_weight = [1,1]
            # class_weights = torch.FloatTensor(self.opt.cross_entropy_weight)
            # self.criterionSegB = torch.nn.CrossEntropyLoss().to(self.device)
            # self.criterionSegB = self.dice_loss().to(self.device)

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            # self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters(), self.netG_A1.parameters(), self.netG_B1.parameters(), self.netS_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999),weight_decay=0.0001)
            # self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters(), self.netD_A1.parameters(), self.netD_B1.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999),weight_decay=0.0001)
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters(), self.netS_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999),weight_decay=0.0001)
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999),weight_decay=0.0001)


        #   self.optimizer_G1 = torch.optim.Adam(itertools.chain(self.netG_A1.parameters(), self.netG_B1.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
        #    self.optimizer_D1 = torch.optim.Adam(itertools.chain(self.netD_A1.parameters(), self.netD_B1.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        # SHUNXING
        # self.real_A1 = input['A1'].to(self.device)
        # self.real_B1 = self.real_B # no need to create a new tensors from unaligned data

        self.Atruth = input['Atruth'].to(self.device)
        self.Btruth = input['Btruth'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

        # SHUNXING
        # self.fake_B1 = self.netG_A1(self.real_A1)  # G_A(A)
        # self.rec_A1 = self.netG_B1(self.fake_B1)   # G_B(G_A(A))
        # self.fake_A1 = self.netG_B1(self.real_B1)  # G_B(B)
        # self.rec_B1 = self.netG_A1(self.fake_A1)   # G_A(G_B(B))
        
        # self.encode_A = F.log_softmax(self.netG_A(self.real_A,True))
        # self.encode_A1 = F.softmax(self.netG_A1(self.real_A1,True))
        # print(self.encode_A.shape)
        # print(self.encode_A1.shape)

        # tmp_fakeB = [self.fake_B +  self.fake_B1]
        # self.avg_fakeB = (self.fake_B + self.fake_B1) / 2
        
        # for segmentation
        self.seg_B_raw_from_S_B = self.netS_B(self.fake_B) # seg(G_A(A))

        _, self.seg_B = torch.max(self.seg_B_raw_from_S_B.data,dim=1,keepdim=True)
        self.seg_B = torch.squeeze(self.seg_B)

        self.seg_A_raw_from_S_A = self.netS_B(self.fake_A) # seg(G_A(A))

        _, self.seg_A = torch.max(self.seg_A_raw_from_S_A.data,dim=1,keepdim=True)
        self.seg_A = torch.squeeze(self.seg_A)

        # x_predict = self.seg_B.detach().to("cpu").numpy()
        # x_truth = self.Atruth.detach().to("cpu").numpy()

        # x_predict = x_predict[0,:,:]
        # x_predict[x_predict == 2] = 127
        # x_predict[x_predict == 1] = 255

        # # print(x_predict.shape)
        # x_truth = x_truth[0,0,:,:]
        # x_truth[x_truth == 2] = 127
        # x_truth[x_truth == 1] = 255

        # x_tmp_list = []
        # x_tmp_list.append(x_truth)
        # x_tmp_list.append(x_predict)
        # x_merge = np.hstack(x_tmp_list)
        # # print(x_truth.shape)
        # cv2.imwrite('x_merge.png', x_merge)
        # print(np.unique(x_predict))
        
        # print(torch.unique(self.seg_b_tmp))
        # print('###########')
        # _, self.seg_B = torch.max(self.seg_b_tmp.data,dim=1,keepdim=True)
        # self.seg_B = torch.squeeze(self.seg_B)
        # print(torch.unique(self.seg_B))

        # print(type(self.avg_fakeB))
        # print(self.avg_fakeB.size())



    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    # def backward_D_A1(self):
    #     """Calculate GAN loss for discriminator D_A1"""
    #     fake_B1 = self.fake_B1_pool.query(self.fake_B1)
    #     self.loss_D_A1 = self.backward_D_basic(self.netD_A1, self.real_B1, fake_B1)

    # def backward_D_B1(self):
    #     """Calculate GAN loss for discriminator D_B1"""
    #     fake_A1 = self.fake_A1_pool.query(self.fake_A1)
    #     self.loss_D_B1 = self.backward_D_basic(self.netD_B1, self.real_A1, fake_A1)


    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_C = self.opt.lambda_C
        # Identity loss
        # SHUNXING: just make sure the L1 loss between fakeB and fakeB1       
        # if lambda_idt > 0:
        #     # G_A should be identity if real_B is fed: ||G_A(B) - B||
        #     self.idt_A = self.netG_A(self.real_B)
        #     self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
        #     # G_B should be identity if real_A is fed: ||G_B(A) - A||
        #     self.idt_B = self.netG_B(self.real_A)
        #     self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        # else:
        #     self.loss_idt_A = 0
        #     self.loss_idt_B = 0

        # self.loss_marker_reproduce_B = self.criterionMarker(self.fake_B, self.fake_B1) #* lambda_A * lambda_idt

        # self.loss_avg_fake_B = self.criterionMarker(self.fake_B, self.avg_fakeB)
        # self.loss_avg_fake_B1 = self.criterionMarker(self.fake_B1, self.avg_fakeB)
        # self.loss_encode_latent = self.criterionEncodeLatentSpace(torch.flatten(self.encode_A), torch.flatten(self.encode_A1)) # latent space KL divergence Loss 
        
        # self.loss_encode_latent = self.criterionEncodeLatentSpace(self.encode_A, self.encode_A1) * lambda_A # 100 #lambda_A # latent space KL divergence Loss 
        
        # SHUNXING - simple trick to convert self.Atruth to category value
        # print(torch.unique(self.Atruth))
        # print(self.Atruth.dtype)
        # tmp_Atruth = self.Atruth
        # tmp_Atruth[tmp_Atruth==-1] = 0
        # tmp_Atruth[tmp_Atruth==1] = 1
        # tmp_Atruth.to(dtype=torch.int64)

        # tmp_Atruth = tmp_Atruth.type(torch.int64)
        # print(torch.unique(tmp_Atruth))
        # print(tmp_Atruth.dtype)

        # print(self.seg_B_raw_from_S_B.shape)
        # print(self.Atruth.shape)
        # print(torch.unique(self.Atruth))
        self.loss_seg_B = self.dice_loss(self.Atruth.to('cpu').long(),self.seg_B_raw_from_S_B.to('cpu')) # .to(self.device)
        # print(self.loss_seg_B)

        self.loss_seg_A = self.dice_loss(self.Btruth.to('cpu').long(),self.seg_A_raw_from_S_A.to('cpu'))

        # self.opt.cross_entropy_weight = [1,1]
        # class_weights = torch.FloatTensor(self.opt.cross_entropy_weight)

        # self.seg_B_raw_from_S_B = self.seg_B_raw_from_S_B.to(self.device)
        # self.loss_seg_B = self.ce_loss(self.Atruth,self.seg_B_raw_from_S_B,class_weights,0) # .to(self.device)

        # self.loss_seg_B = self.criterionSegB(self.seg_B_raw_from_S_B,self.Atruth)


        # m = torch.nn.Sigmoid()
        # self.loss_seg_B = self.criterionSegB(m(self.seg_B),self.Atruth) * lambda_C
        # print(self.loss_seg_B )
        # self.loss_seg_B = self.criterionSegB(self.seg_B,tmp_Atruth) * lambda_C
        # print(self.loss_encode_latent)
        # print(self.loss_encode_latent)
        
        # self.loss_encode_latent = self.criterionEncodeLatentSpace(self.encode_A, self.encode_A1) # latent space KL divergence Loss 

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        #self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B



        # lambda_A1 = self.opt.lambda_A
        # lambda_B1 = self.opt.lambda_B

        # # self.loss_marker_reproduce_B1 = self.criterionMarker(self.fake_B1, self.fake_B) # * lambda_A * lambda_idt

        # # GAN loss D_A(G_A(A))
        # self.loss_G_A1 = self.criterionGAN(self.netD_A1(self.fake_B1), True)
        # # GAN loss D_B(G_B(B))
        # self.loss_G_B1 = self.criterionGAN(self.netD_B1(self.fake_A1), True)
        # # Forward cycle loss || G_B(G_A(A)) - A||
        # self.loss_cycle_A1 = self.criterionCycle(self.rec_A1, self.real_A1) * lambda_A1
        # # Backward cycle loss || G_A(G_B(B)) - B||
        # self.loss_cycle_B1 = self.criterionCycle(self.rec_B1, self.real_B1) * lambda_B1

        
        # self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_marker_reproduce_B +self.loss_G_A1 +  self.loss_G_B1 + self.loss_cycle_A1 + self.loss_cycle_B1 + self.loss_encode_latent
        # remove KL and see what's going on
        #self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_marker_reproduce_B +self.loss_G_A1 +  self.loss_G_B1 + self.loss_cycle_A1 + self.loss_cycle_B1  + self.loss_encode_latent + self.loss_avg_fake_B + self.loss_avg_fake_B1
        # self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_marker_reproduce_B +self.loss_G_A1 +  self.loss_G_B1 + self.loss_cycle_A1 + self.loss_cycle_B1   + self.loss_seg_B
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B  + self.loss_seg_B + self.loss_seg_A
        
        self.loss_G.backward()
        

#        lambda_A1 = self.opt.lambda_A
#        lambda_B1 = self.opt.lambda_B

        # self.loss_marker_reproduce_B1 = self.criterionMarker(self.fake_B1, self.fake_B) # * lambda_A * lambda_idt

        # GAN loss D_A(G_A(A))
#        self.loss_G_A1 = self.criterionGAN(self.netD_A1(self.fake_B1), True)
        # GAN loss D_B(G_B(B))
#        self.loss_G_B1 = self.criterionGAN(self.netD_B1(self.fake_A1), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
#        self.loss_cycle_A1 = self.criterionCycle(self.rec_A1, self.real_A1) * lambda_A1
        # Backward cycle loss || G_A(G_B(B)) - B||
#        self.loss_cycle_B1 = self.criterionCycle(self.rec_B1, self.real_B1) * lambda_B1
        # combined loss and calculate gradients
        #self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
#        self.loss_G1 = self.loss_G_A1 + self.loss_G_B1 + self.loss_cycle_A1 + self.loss_cycle_B1 + self.loss_marker_reproduce_B1

#        self.loss_G1.backward()

    # def backward_G1(self):
    #     """Calculate the loss for generators G_A and G_B"""
    #     lambda_idt = self.opt.lambda_identity
    #     lambda_A1 = self.opt.lambda_A
    #     lambda_B1 = self.opt.lambda_B

    #     self.loss_marker_reproduce_B1 = self.criterionMarker(self.fake_B1, self.fake_B) # * lambda_A * lambda_idt

    #     # GAN loss D_A(G_A(A))
    #     self.loss_G_A1 = self.criterionGAN(self.netD_A1(self.fake_B1), True)
    #     # GAN loss D_B(G_B(B))
    #     self.loss_G_B1 = self.criterionGAN(self.netD_B1(self.fake_A1), True)
    #     # Forward cycle loss || G_B(G_A(A)) - A||
    #     self.loss_cycle_A1 = self.criterionCycle(self.rec_A1, self.real_A1) * lambda_A1
    #     # Backward cycle loss || G_A(G_B(B)) - B||
    #     self.loss_cycle_B1 = self.criterionCycle(self.rec_B1, self.real_B1) * lambda_B1
    #     # combined loss and calculate gradients
    #     #self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
    #     self.loss_G1 = self.loss_G_A1 + self.loss_G_B1 + self.loss_cycle_A1 + self.loss_cycle_B1 + self.loss_marker_reproduce_B1
    #     self.loss_G1.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        # self.set_requires_grad([self.netD_A, self.netD_B, self.netD_A1, self.netD_B1], False)  # Ds require no gradients when optimizing Gs
        
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        

        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights


        # # G_A1 and G_B1
        # self.set_requires_grad([self.netD_A1, self.netD_B1], False)  # Ds require no gradients when optimizing Gs
        # self.optimizer_G1.zero_grad()  # set G_A and G_B's gradients to zero
        # self.backward_G1()             # calculate gradients for G_A and G_B
        # self.optimizer_G1.step()       # update G_A and G_B's weights


        # D_A and D_B
        # self.set_requires_grad([self.netD_A, self.netD_B, self.netD_A1, self.netD_B1], True)
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        # self.backward_D_A1()
        # self.backward_D_B1()
        self.optimizer_D.step()  # update D_A and D_B's weights



        # D_A1 and D_B1
    #    self.set_requires_grad([self.netD_A1, self.netD_B1], True)
    #    self.optimizer_D1.zero_grad()   # set D_A and D_B's gradients to zero
    #    self.backward_D_A1()      # calculate gradients for D_A
    #    self.backward_D_B1()      # calculate graidents for D_B
    #    self.optimizer_D1.step()  # update D_A and D_B's weights
    # def dice_loss(input, target):
    #     smooth = 1.
    #     loss = 0.
    #     for c in range(n_classes):
    #         iflat = input[:, c ].view(-1)
    #         tflat = target[:, c].view(-1)
    #         intersection = (iflat * tflat).sum()
            
    #         w = class_weights[c]
    #         loss += w*(1 - ((2. * intersection + smooth) /
    #                             (iflat.sum() + tflat.sum() + smooth)))
    #     return loss

    def dice_loss(self, true, logits, eps=1e-7):
        """Computes the Sørensen–Dice loss.
        Note that PyTorch optimizers minimize a loss. In this
        case, we would like to maximize the dice loss so we
        return the negated dice loss.
        Args:
            true: a tensor of shape [B, 1, H, W].
            logits: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
            eps: added to the denominator for numerical stability.
        Returns:
            dice_loss: the Sørensen–Dice loss.
        """
        # print(logits.shape)
        # print(true.shape)
        num_classes = logits.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            # print(true)
            # print(type(true))
            # print(true.shape)
            # print('####')
            # print(logits.shape)
            # num_classes = num_classes.to(self.device)
            # x1 = true.squeeze(1)
            # print(x1)
            true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + eps)).mean()
        return (1 - dice_loss)

    def ce_loss(self, true, logits, weights, ignore=255):
        """Computes the weighted multi-class cross-entropy loss.
        Args:
            true: a tensor of shape [B, 1, H, W].
            logits: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
            weight: a tensor of shape [C,]. The weights attributed
                to each class.
            ignore: the class index to ignore.
        Returns:
            ce_loss: the weighted multi-class cross-entropy loss.
        """
        ce_loss = F.cross_entropy(
            logits.float(),
            true.long(),
            ignore_index=ignore,
            weight=weights,
        )
        return ce_loss
