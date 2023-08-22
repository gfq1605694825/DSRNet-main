import os

import skimage.io as io
import torch
import torch.nn.functional as F
from data import generate_loader
from tqdm import tqdm


class Tester():
    def __init__(self, module, opt):
        self.opt = opt

        self.dev = torch.device("cuda:{}".format(opt.GPU_ID) if torch.cuda.is_available() else "cpu")
        self.net = module.Net(opt)
        self.net = self.net.to(self.dev)

        msg = "# params:{}\n".format(
            sum(map(lambda x: x.numel(), self.net.parameters())))
        print(msg)

    @torch.no_grad()
    def evaluate(self, path):
        opt = self.opt
        try:
            print('loading model from: {}'.format(path))
            self.load(path)
        except Exception as e:
            print(e)

        self.net.eval()

        if opt.save_result:
            save_root = os.path.join(opt.save_root, opt.save_msg)
            os.makedirs(save_root, exist_ok=True)

        test_paths = opt.test_paths.split('+')
        torch.cuda.manual_seed_all(1)
        for test_dir_img in test_paths:
            print(opt.test_save_path + test_dir_img + '/')
            if not os.path.exists(opt.test_save_path + test_dir_img):
                os.makedirs(opt.test_save_path + test_dir_img + '/', exist_ok=True)

            opt.test_dataset = "dsrnet_" + test_dir_img
            print(opt.test_dataset)
            test_loader = generate_loader("test", opt)

            for i, inputs in enumerate(tqdm(test_loader)):
                MASK = inputs[0].to(self.dev)
                IMG = inputs[1].to(self.dev)
                NAME = inputs[2][0]

                b, c, h, w = MASK.shape

                pred = self.net(IMG)

                # Save the last stage of the output upsampling reduction channel
                pred = F.pixel_shuffle(pred[7], 4)

                pred = F.interpolate(pred, (h, w), mode='bilinear', align_corners=False)

                pred = torch.sigmoid(pred).squeeze()

                pred = (pred * 255.).detach().cpu().numpy().astype('uint8')

                if opt.save_result:

                    if opt.save_all:
                        for idx, sal in enumerate(pred[1:]):
                            scale = 224 // (sal.shape[-1])
                            sal_img = F.pixel_shuffle(sal, scale)
                            sal_img = F.interpolate(sal_img, (h, w), mode='bilinear', align_corners=False)
                            sal_img = torch.sigmoid(sal_img)
                            sal_path = os.path.join(save_root, "{}_sal_{}.png".format(NAME, idx))
                            sal_img = sal_img.squeeze().detach().cpu().numpy()
                            sal_img = (sal_img * 255).astype('uint8')
                            io.imsave(sal_path, sal_img)
                    else:
                        # save pred image
                        save_path_sal = opt.test_save_path + test_dir_img + '/'

                        savePath = os.path.join(save_path_sal, "{}.png".format(NAME))

                        io.imsave(savePath, pred, check_contrast=False)
        return 0

    def load(self, path):
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(state_dict)
        return
