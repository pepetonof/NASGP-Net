import torch
import torch.nn as nn
# import torchvision.transforms.functionalJ as TF
import torchvision.transforms.functional as TF
from numpy import sqrt
import copy

from operators.seBlock import SELayer
from operators.moduleConv import moduleconv
from operators.functionSet import MyModule_AddSubCat
from operators.ResConnection import ResBlockConnection
from operators.DenseConnection import DenseBlockConnection

#%%For blocks construction
class Blocks():
    def __init__(self, first_block):
        self.first_block=first_block
        # self.first_block=first_block
        # self.downs = self.generate_blocks(first_block, 4)
        #elf.bottleneck, self.ups
        
    def change_in_out(self, mod, channels):
        mod.in_channels=channels
        mod.out_channels=mod.out_channels*2
        mod.groupsGN=mod.groupsGN*2
        mod.make_layer(type=mod.type)
        
    def double(self, modulelist, out_channels):
        for layer in modulelist:
            if isinstance(layer, moduleconv):
                self.change_in_out(layer, out_channels)
                out_channels=layer.out_channels
            
            if isinstance(layer, ResBlockConnection):
                seq=layer.moduleList
                for l in seq:
                    self.change_in_out(l, out_channels)
                    out_channels=l.out_channels 
                layer.make_connection(copy.deepcopy(layer.moduleList))
                out_channels=layer.out_channels
                
            if isinstance(layer, DenseBlockConnection):
                seq=layer.moduleList
                for l in seq:
                    self.change_in_out(l, out_channels)
                    out_channels=l.out_channels
                layer.make_connection(copy.deepcopy(layer.moduleList), layer.tetha)
                out_channels=layer.out_channels
            
            if isinstance(layer, SELayer):
                layer.out_channels=out_channels
                layer.make_layer(layer.out_channels, layer.reduction)
            
            if isinstance(layer, MyModule_AddSubCat):
                flag=layer.flag
                oc=[]
                for idx, seq in enumerate(layer.moduleList):#2sequences
                    # print('double', out_channels)
                    _, out_channels_aux=self.double(seq, out_channels)
                    # print('double', out_channels_aux, '\n')
                    oc.append(out_channels_aux)
                if flag==2:
                    out_channels=oc[0]+oc[1]
                else:
                    out_channels=max(oc)
                # print(out_channels)
                                                   
        return modulelist, out_channels
                    
    def down_blocks(self, b, n):
        dblocks=[b]
        out_channels=b.out_channels
        for i in range(1, n):
            bc=copy.deepcopy(dblocks[i-1])
            _, out_channels=self.double(bc.moduleListCell[:-1], out_channels)
            dblocks.append(bc)
            bc.out_channels = out_channels

        #last element for bottleneck
        bc=copy.deepcopy(dblocks[-1])
        _, out_channels = self.double(bc.moduleListCell[:-1], out_channels)
        dblocks.append(bc)
        bc.out_channels = out_channels
            
        return dblocks
    
    
    def replace(self, replace, in_channels, out_channels, kernel_size=2, stride=2):
        layer=replace(in_channels, out_channels, kernel_size, stride)
        return layer
    
    def change_in(self, modulelist, in_channels):
        
        def change_in_aux(mod, in_channels):
            mod.in_channels=in_channels
            mod.make_layer(mod.type)
            
        # print(modulelist, in_channels)
        for layer in modulelist:
            if isinstance(layer, moduleconv):
                # print(layer, in_channels, '\n')
                change_in_aux(layer, in_channels)
                # out_channels=layer.out_channels
                out_channels=modulelist[-1].out_channels#!!!
                break
                
            if isinstance(layer, ResBlockConnection):
                change_in_aux(layer.moduleList[0], in_channels)
                layer.make_connection(layer.moduleList)
                out_channels=layer.out_channels
            
            if isinstance(layer, DenseBlockConnection):
                change_in_aux(layer.moduleList[0], in_channels)
                layer.make_connection(copy.deepcopy(layer.moduleList), layer.tetha)
                out_channels=layer.out_channels
            
            if isinstance(layer, SELayer):
                layer.out_channels=out_channels
                layer.make_layer(layer.out_channels, layer.reduction)
                
            if isinstance(layer, MyModule_AddSubCat):
                # print(layer, in_channels,'\n')
                flag=layer.flag
                oc=[]
                for seq in layer.moduleList:#2 sequences
                    # print('ChangeIn', seq)
                    out_channels_aux=self.change_in(seq, in_channels)
                    # print('ChangeIn', out_channels_aux)
                    oc.append(out_channels_aux)
                if flag==2:
                    out_channels=oc[0]+oc[1]
                else:
                    out_channels=max(oc)
                # print('ChangeIn', oc)
                # layer.out_channels=out_channels
                # print(layer.out_channels)
                
        return out_channels
                
    def up_blocks(self, dblocks):
        features=[]
        for i in range(len(dblocks[:-1])):
            features.append(dblocks[i].out_channels)
        features.reverse()
        # print(features)
        
        ublocks=list(reversed(copy.deepcopy(dblocks[:-1])))
        ublocks.insert(0, dblocks[-1])
        
        #Change in_channels of the four next blocks
        # in_channels=features[0]
        for (feat, block) in zip(features, ublocks[1:]):
            in_channels=feat
            out_channels=self.change_in(block.moduleListCell[:-1], in_channels*2)
            
            block.out_channels=out_channels
            # in_channels=feat
            # print(block)
            # print(out_channels, block.out_channels)
        
        #Replace every pooling operation from bottleneck to last one
        for (feat, block) in zip(features, ublocks[:-1]):
            # print(block, block.out_channels, feat)
            block.moduleListCell[-1]=self.replace(nn.ConvTranspose2d, block.out_channels, feat)
            # print(block, block.out_channels, feat, '\n\n\n')
            
        
        #Last block
        ublocks[-1].moduleListCell[-1]=self.replace(nn.Conv2d, ublocks[-1].out_channels, 2, 1, 1)
        # print(ublocks[-1], ublocks[-1].out_channels)
        
        return ublocks
    
    def generate_cells(self, b1, num_blocks):
        dblocks = self.down_blocks(b1, num_blocks)
        ublocks = self.up_blocks(dblocks)
        # print('DBLOCKS',dblocks,'\n')
        # print('UBLOCKS',ublocks)
        
        # print(len(dblocks), len(ublocks))
        down_blocks = nn.ModuleList(dblocks[:-1])
        bottleneck  = ublocks[0]
        up_blocks   = nn.ModuleList(ublocks[1:])
        
        # print(up_blocks[0])
        
        return down_blocks, bottleneck, up_blocks
        
        

#%%BackBone UNET
class BackBone(nn.Module):
    def __init__(self, first_block):
        super(BackBone, self).__init__()

        #Blocks of backbone
        backbone = Blocks(first_block)
        self.downs, self.bottleneck, self.ups = backbone.generate_cells(first_block, 4)

        #Dropout of 50%
        self.dropout = nn.Dropout(p=0.5)
        
        #Change in_channels for depthwise separable convolution
        self.set_ch_rec(self.downs)
        self.set_ch_rec(self.bottleneck)
        self.set_ch_rec(self.ups)
        
        #Initialize weigths
        self.initialize_weigths()

    def groupsFixed(self, outChannels, init_groups):
        #print('GroupsFixed', outChannels, init_groups)
        div=[]
        if outChannels%init_groups==0:
            return init_groups
        elif outChannels==1:
            return 1 
        else:
            for i in range(2, outChannels+1):
                if outChannels%i==0:
                    div.append(i)
            return min(div)
    
    #Change in_channels for depthwise separable convolution
    def set_ch_rec(self, module):    
        for n, m in module.named_children():
            #compund module
            if len(list(m.children()))>0:
                self.set_ch_rec(m)
            #convolution
            if n =='depthwise':
                
                num_groups=m[1].num_groups
                
                groups=m[0].in_channels
                in_channels=m[0].in_channels
                # print('in_channelsDWS',in_channels)
                kernel_size=m[0].kernel_size
                dilation_rate=m[0].dilation
                
                # print(in_channels, kernel_size, dilation_rate, num_groups)
                 
                num_groups=self.groupsFixed(in_channels, num_groups)
                
                # print(num_groups)
                new=nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size,
                                            stride=1, padding='same', dilation=dilation_rate, 
                                            groups=groups, bias=False),
                                            # nn.BatchNorm2d(in_channels),
                                            nn.GroupNorm(num_groups,in_channels),
                                            nn.ReLU(inplace=True)
                                            )
                setattr(module, n, new)
            if n=='pointwise':
                #in_channels=m[0].in_channels#
                num_groups=m[1].num_groups
                
                kernel_size=m[0].kernel_size
                dilation_rate=m[0].dilation
                out_channels=m[0].out_channels
                # print('out_channelsDWS',out_channels)
                
                new=nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size,
                                            stride=1, padding='same', dilation=dilation_rate, 
                                            groups=1, bias=False),
                                            # nn.BatchNorm2d(out_channels),
                                            nn.GroupNorm(num_groups,out_channels),
                                            nn.ReLU(inplace=True)
                                            )
                setattr(module, n, new)
            if n=='pointwiseRes':
                # in_channels=m[0].in_channels#!!!
                
                num_groups=m[1].num_groups
                
                kernel_size=m[0].kernel_size
                dilation_rate=m[0].dilation
                out_channels=m[0].out_channels
                # print('out_channelsDWS',out_channels)
                new=nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size,
                                            stride=1, padding='same', dilation=dilation_rate, 
                                            groups=1, bias=False),
                                            # nn.BatchNorm2d(out_channels),
                                            nn.GroupNorm(num_groups,out_channels),
                                            )
                setattr(module, n, new)
        return
        
    #Initialize_weigths accordintg to Ronnenberger 
    def initialize_weigths(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.in_channels!=0:
                
                # print(m)
                # print(m.kernel_size, m.in_channels)
                # if m.in_channels==0:
                    # print(m,'error')
                std=sqrt(2/(m.kernel_size[0]*m.kernel_size[1]*m.in_channels))
                nn.init.normal_(m.weight, std=std)
    
    def forward(self, x):
        skip_connections = []
        # print('ForwardDown:\t')
        
        for down in self.downs:
            # print('block', down)
            # print('down',x.shape)
            x = down.forwardpart(x, 'first')
            # print('Down1\t', x.shape)
            skip_connections.append(x)
            x = down.forwardpart(x, 'second')
            # print('Down2\t', x.shape)
        # print('Downs', x.shape)
        
        # #Apply 1st dropout
        x = self.dropout(x)
        #Bottleneck
        x = self.bottleneck(x)
        #Apply 2nd dropout
        x = self.dropout(x)
        # print('\nBottle:\t', x.shape)
        
        #Reverse list
        # print(len(skip_connections))
        skip_connections = skip_connections[::-1]
        
        # print('\nForwardUp:\t')
        for idx in range(len(self.ups)):
            # print('in_shape', x.shape)
            skip_connection = skip_connections[idx]
            # print('skip:\t', skip_connection.shape, x.shape)
            if x.shape[2:] != skip_connection.shape[2:]:
                # print('reshaping...')
                #skip_connection = TF.resize(skip_connection, size=x.shape[2:])
                #x = TF.resize(x, size=skip_connection.shape[2:])
                #x = TF.center_crop(x, output_size=skip_connection.shape[2:])
                skip_connection = TF.center_crop(skip_connection, output_size=x.shape[2:])
                # print('reshape skip', skip_connection.shape, x.shape) 
            
            # if skip_connection.shape==x.shape:
            #     concat_skip = skip_connection + x
            # else:
            #     concat_skip = torch.cat((skip_connection, x), dim=1)
            
            concat_skip=torch.cat((skip_connection, x), dim=1)
            
            # concat_skip = skip_connection+x
            # print('concat:\t', concat_skip.shape)
            x = self.ups[idx](concat_skip)
            # print('Up block:\t', x.shape)

        return x
