from retrain_model.new_model import get_default_net
from retrain_model.aspp import ASPP
from retrain_model.decoder import Decoder
import torch.nn as nn
from operations import NaiveBN, ABN


class Retrain_Autodeeplab(nn.Module):
    def __init__(self, args):
        super(Retrain_Autodeeplab, self).__init__()
        BatchNorm2d = ABN if args.use_ABN else NaiveBN
        if args.use_ABN:
            print("=> use ABN!")
        self.encoder = get_default_net(args=args)
        self.aspp = ASPP(args.filter_multiplier * 10, 256, args.num_classes, conv=nn.Conv2d, norm=BatchNorm2d)
        self.decoder = Decoder(args.num_classes, filter_multiplier=args.filter_multiplier * args.block_multiplier)

    def forward(self, x):
        encoder_output, low_level_feature = self.encoder(x)
        high_level_feature = self.aspp(encoder_output)
        decoder_output = self.decoder(encoder_output, low_level_feature)
        return nn.Upsample((x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)(decoder_output)
