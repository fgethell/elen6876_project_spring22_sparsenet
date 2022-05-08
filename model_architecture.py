from utils import *

class Model(nn.Module):
    '''
    This is the master class initializing the entire neural network model as a torch object. Input parameters for the constructor are defined as follows:
    growth_rate: growth rate value
    depth: depth value of the model
    reduction: reduction value
    num_classes: number of classes
    bottle_neck_flag: flag for bottle neck compression layer
    layers_per_stage: number of layers per dense stage
    growth_rate_per_stage: growth rate value
    drop_prob: 0.0: dropout probability value,
    fetch_type: architecture type (sparse or dense)
    '''
    def __init__(self, 
                 growth_rate, 
                 depth, 
                 reduction, 
                 num_classes, 
                 bottle_neck_flag, 
                 layers_per_stage = None, 
                 growth_rate_per_stage = None,
                 drop_prob = 0.0,
                 fetch_type = "sparse"):
        
        super(Model, self).__init__()

        num_dense_blocks = (depth - 4) // 3 #Compute the number of dense blocks based on the depth
        
        if bottle_neck_flag:
            num_dense_blocks =  num_dense_blocks // 2
            
        else:
            reduction = 1
            
        if layers_per_stage is None:
            layers_per_stage = [num_dense_blocks, ] * 3
            
        if growth_rate_per_stage is None:
            growth_rate_per_stage = [growth_rate, ] * 3

        if fetch_type == "dense":
            num_channels = 2 * growth_rate
            
        elif fetch_type == "sparse":
            num_channels = growth_rate
            
        else:
            raise NotImplementedError

        #Create first kernel layer
        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(3, num_channels, kernel_size = 3, padding = 1, bias = False)),
        ]))

        #Create dense layers equal to the number of layers per stage
        for i, layers in enumerate(layers_per_stage):
            stage = dense_stage(layers_per_stage[i], num_channels, growth_rate_per_stage[i], 
                                bottle_neck_flag, drop_prob = drop_prob, fetch_type = fetch_type)
            
            num_channels = stage.num_output_channels
            
            self.features.add_module("dense-stage-%d" % i, stage)
            
            if i < len(layers_per_stage) - 1:
                
                num_output_channels = int(math.floor(num_channels * reduction))
                
                transition = transition_layer(num_channels, num_output_channels)
                
                self.features.add_module("transition-%d" % i, transition)
                
                num_channels = num_output_channels
                
        self.features.add_module("batch_norm_last_layer", nn.BatchNorm2d(num_channels))
        self.features.add_module("relu_last_layer", nn.ReLU(inplace = True))

        #Create classification layer
        self.classifier = nn.Sequential(OrderedDict([
            ("avgpool", nn.AvgPool2d(kernel_size = 8)),
            ("flattern", flatten_layer(dim = 0)),
            ("linear", nn.Linear(num_channels, num_classes))
        ]))
        
        #Initializing weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        
        features = self.features(x)
        out = self.classifier(features)
        
        return out