"""
Lighter U-net implementation that achieves same performance as the one reported in the paper: https://arxiv.org/abs/1505.04597
Main differences:
    a) U-net downblock has only 1 convolution instead of 2
    b) U-net upblock has only 1 convolution instead of 3
"""

import torch
import torch.nn as nn

from base_temporal_model import BaseTemporalModel

class UNetDownBlock(nn.Module):
    """
    Constructs a UNet downsampling block

       Parameters:
            input_nc (int)      -- the number of input channels
            output_nc (int)     -- the number of output channels
            norm_layer (str)    -- normalization layer
            down_type (str)     -- if we should use strided convolution or maxpool for reducing the feature map
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            user_dropout (bool) -- if use dropout layers.
            kernel_size (int)   -- convolution kernel size
            bias (boolean)      -- if convolution should use bias
    """
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, down_type='strideconv', outermost=False, innermost=False, dropout=0.2, kernel_size=4, bias=True):
        super(UNetDownBlock, self).__init__()
        self.innermost = innermost
        self.outermost = outermost
        self.use_maxpool = down_type == 'maxpool'

        stride = 1 if self.use_maxpool else 2
        kernel_size = 3 if self.use_maxpool else 4
        self.conv = nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride, padding=1, bias=bias)
        self.relu = nn.LeakyReLU(0.2, True)
        self.maxpool = nn.MaxPool2d(2)
        self.norm = norm_layer(output_nc)
        self.dropout = nn.Dropout2d(dropout) if dropout else None

    def forward(self, x):
        if self.outermost:
            x = self.conv(x)
            x = self.norm(x)
        elif self.innermost:
            x = self.relu(x)
            if self.dropout: x = self.dropout(x)
            x = self.conv(x)
        else:
            x = self.relu(x)
            if self.dropout: x = self.dropout(x)
            x = self.conv(x)
            x = self.norm(x)

        return x

class UNetUpBlock(nn.Module):
    """
      Constructs a UNet upsampling block

         Parameters:
              input_nc (int)      -- the number of input channels
              output_nc (int)     -- the number of output channels
              norm_layer          -- normalization layer
              outermost (bool)    -- if this module is the outermost module
              innermost (bool)    -- if this module is the innermost module
              user_dropout (bool) -- if use dropout layers.
              kernel_size (int)   -- convolution kernel size
              remove_skip (bool)  -- if skip connections should be disabled or not
      """
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, outermost=False, innermost=False, dropout=0.2, kernel_size=4, remove_skip=0, use_bias=True):
        super(UNetUpBlock, self).__init__()
        self.innermost = innermost
        self.outermost = outermost
        self.remove_skip = remove_skip
        upconv_inner_nc = input_nc if self.remove_skip else input_nc * 2

        if self.innermost:
            self.conv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=kernel_size, stride=2, padding=1, bias=use_bias)
        elif self.outermost:
            self.conv = nn.ConvTranspose2d(upconv_inner_nc, output_nc, kernel_size=kernel_size, stride=2, padding=1)
        else:
            self.conv = nn.ConvTranspose2d(upconv_inner_nc, output_nc, kernel_size=kernel_size, stride=2, padding=1, bias=use_bias)

        self.norm = norm_layer(output_nc)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout) if dropout else None

    def forward(self, x):
        if self.outermost:
            x = self.relu(x)
            if self.dropout: x = self.dropout(x)
            x = self.conv(x)
        elif self.innermost:
            x = self.relu(x)
            if self.dropout: x = self.dropout(x)
            x = self.conv(x)
            x = self.norm(x)
        else:
            x = self.relu(x)
            if self.dropout: x = self.dropout(x)
            x = self.conv(x)
            x = self.norm(x)

        return x

class UNet(BaseTemporalModel):
    """Create a Unet-based Fully Convolutional Network
          X -------------------identity----------------------
          |-- downsampling -- |submodule| -- upsampling --|

        Parameters:
            num_classes (int)      -- the number of channels in output images
            norm_layer             -- normalization layer
            input_nc               -- number of channels of input image

            Args:
            mode (str)             -- process single frames or sequence of frames
            timesteps (int)        --
            num_downs (int)        -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                      image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)              -- the number of filters in the last conv layer
            remove_skip (int [0,1])-- if skip connections should be disabled or not
            reconstruct (int [0,1])-- if we should reconstruct the next image or not
            sequence_model (str)   -- the sequence model that for the sequence mode []
            num_levels_tcn(int)    -- number of levels of the TemporalConvNet
      """

    def __init__(self, num_classes, args, norm_layer=nn.BatchNorm2d, input_nc=3):
        super(UNet, self).__init__(args)

        self.num_downs = args.num_downs
        self.ngf = args.ngf
        self.remove_skip = args.remove_skip
        self.segmentation = args.segmentation
        self.reconstruct = args.reconstruct
        self.reconstruct_remove_skip = args.reconstruct_remove_skip

        if args.mode == 'fbf-1234':
            input_nc = input_nc*4
        self.encoder = self.build_encoder(self.num_downs, input_nc, self.ngf, norm_layer, down_type=args.down_type)

        self.skip_sequence_models = self.get_skip_sequence_models(args) if '+temporal_skip' in self.sequence_model_type else None
        self.encoder_sequence_models = self.get_skip_sequence_models(args) if '+temporal_encoder' in self.sequence_model_type else None
        self.all_sequence_models = self.get_skip_sequence_models(args) if '+temporal_all' in self.sequence_model_type else None
        self.sequence_model = self.get_sequence_model(args) if 'sequence' in self.mode else None

        if args.segmentation:
            self.decoder = self.build_decoder(self.num_downs, num_classes, self.ngf, norm_layer, remove_skip=self.remove_skip)

        if self.reconstruct:
            self.reconstruction_decoder = self.build_decoder(self.num_downs, input_nc, self.ngf, norm_layer, remove_skip=args.reconstruct_remove_skip)

    def build_encoder(self, num_downs, input_nc, ngf, norm_layer, down_type='strideconv'):
        """Constructs a UNet downsampling encoder, consisting of $num_downs UNetDownBlocks

             Parameters:
                  num_downs (int)     -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                         image of size 128x128 will become of size 1x1 # at the bottleneck
                  input_nc (int)      -- the number of input channels
                  ngf (int)           -- the number of filters in the last conv layer
                  norm_layer (str)    -- normalization layer
                  down_type (str)     -- if we should use strided convolution or maxpool for reducing the feature map
             Returns:
                  nn.Sequential consisting of $num_downs UnetDownBlocks
        """
        layers = []
        layers.append(UNetDownBlock(input_nc=input_nc, output_nc=ngf, norm_layer=norm_layer, down_type=down_type, outermost=True))
        layers.append(UNetDownBlock(input_nc=ngf, output_nc=ngf*2, norm_layer=norm_layer, down_type=down_type))
        layers.append(UNetDownBlock(input_nc=ngf*2, output_nc=ngf*4, norm_layer=norm_layer, down_type=down_type))
        layers.append(UNetDownBlock(input_nc=ngf*4, output_nc=ngf*8, norm_layer=norm_layer, down_type=down_type))

        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            layers.append(UNetDownBlock(input_nc=ngf*8, output_nc=ngf*8, norm_layer=norm_layer, down_type=down_type))

        layers.append(UNetDownBlock(input_nc=ngf*8, output_nc=ngf*8, norm_layer=norm_layer, down_type=down_type, innermost=True))

        return nn.Sequential(*layers)

    def build_decoder(self, num_downs, num_classes, ngf, norm_layer, remove_skip=0):
        """Constructs a UNet downsampling encoder, consisting of $num_downs UNetUpBlocks

           Parameters:
                num_downs (int)     -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                       image of size 128x128 will become of size 1x1 # at the bottleneck
                num_classes (int)   -- number of classes to classify
                output_nc (int)     -- the number of output channels. outermost is ngf, innermost is ngf * 8
                norm_layer          -- normalization layer
                remove_skip (int)   -- if skip connections should be disabled or not

           Returns:
                nn.Sequential consisting of $num_downs UnetUpBlocks
        """
        layers = []
        layers.append(UNetUpBlock(input_nc=ngf * 8, output_nc=ngf * 8, norm_layer=norm_layer, remove_skip=remove_skip, innermost=True))

        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            layers.append(UNetUpBlock(input_nc=ngf * 8, output_nc=ngf * 8, norm_layer=norm_layer, remove_skip=remove_skip))

        layers.append(UNetUpBlock(input_nc=ngf * 8, output_nc=ngf * 4, norm_layer=norm_layer, remove_skip=remove_skip))
        layers.append(UNetUpBlock(input_nc=ngf * 4, output_nc=ngf * 2, norm_layer=norm_layer, remove_skip=remove_skip))
        layers.append(UNetUpBlock(input_nc=ngf*2, output_nc=ngf, norm_layer=norm_layer, remove_skip=remove_skip))
        layers.append(UNetUpBlock(input_nc=ngf, output_nc=num_classes, norm_layer=norm_layer, remove_skip=remove_skip, outermost=True))

        return nn.Sequential(*layers)

    def encoder_forward(self, x):
        skip_connections = []

        for i, down in enumerate(self.encoder):
            x = down(x)
            if down.use_maxpool:
                x = down.maxpool(x)

            if self.encoder_sequence_models and not down.innermost:
                x = self.temporal_forward(x, self.encoder_sequence_models[i])
            elif self.all_sequence_models and not down.innermost:
                x = self.temporal_forward(x, self.all_sequence_models[i])

            if not down.innermost:
                skip_connections.append(x)

        return x, skip_connections

    def decoder_forward(self, x, skip_connections):
        for i, up in enumerate(self.decoder):
            if not up.innermost:
                if self.all_sequence_models:
                    out = self.temporal_forward(out, self.all_sequence_models[-i])

                if not self.remove_skip:
                    skip = skip_connections[-i]
                    out = torch.cat([skip, out], 1)
                out = up(out)
            else:
                out = up(x)

        return out

    def reconstruction_decoder_forward(self, x, skip_connections):
        for i, up in enumerate(self.reconstruction_decoder):
            if not up.innermost:
                if not self.reconstruct_remove_skip:
                    skip = skip_connections[-i]
                    out = torch.cat([skip, out], 1)
                out = up(out)
            else:
                out = up(x)

        return out

    def forward(self, input):
        x = self.remove_time_reshape(input)
        x, skip_connections = self.encoder_forward(x)

        if self.sequence_model:
            x = self.temporal_forward(x, self.sequence_model)

        if self.skip_sequence_models:
            skip_connections = self.skip_connection_temporal_forward(skip_connections)

        reconstruction_output = self.reconstruction_decoder_forward(x, skip_connections) if self.reconstruct else None
        segmentation_output = self.decoder_forward(x, skip_connections) if self.segmentation else None

        if 'sequence' in self.mode:
            segmentation_output = self.add_time_reshape(segmentation_output) if self.segmentation else None
            reconstruction_output = self.add_time_reshape(reconstruction_output) if self.reconstruct else None

        return segmentation_output, reconstruction_output
