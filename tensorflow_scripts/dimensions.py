'''

This file is me and david trying to understand what is going on ??()(?>??)

input image size: (4, 16, 64, 64, 1)
this is because we have 2d slices of 64 x 64 and there are 16 slices. we take batch sizes of 4 and they are gray scale

we do a conv32 with a filter size of [5,5,5,1,32]

In tensor flow we define filter = [kernel_depth, kernel_height, kernel_width, in_channels, out_channels]

so the dimensions of the output image become = [ 4, 16, 64, 64, 32 ]

We then do a resnet_block with filter size [3,3,3,32,32]
so image output size becomes = [ 4, 16, 64, 64, 32 ]

In a resnet_block the number of input channels == number of output channels. 
This is because you have to add it within the input, so they have to be the same size. 


CONV3D --> RESNET --> MAXPOOL --> CONV3 
it continues doing this until the number of channels of the image is 256 

then there are some deep layers or resnets

then we begin with the decoder ( up the other side of the UNet )
the filter in the decoder is defined in the opposite way = [depth,height, width, output_channels, in_channels]

we then  do blocks of CONV£D_TRANSPOSE -->RESNET 

until we reach the image size again of (4, 16, 64, 64, 1). Then we pass all this through a sigmoid.
A sigmoid function gives us a probability of each pixel being within the prostate or not. (since we only have two classes)


'''