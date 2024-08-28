from torch import nn


class FTModel(nn.Module):
    def __init__(self, pretrained, layers_to_remove, num_features, num_classes,
                 input_size=3, train_only_fc=False):
        super(FTModel, self).__init__()

        # extract number of features in the last layer before the ones we remove
        in_features = list(pretrained.children())[-layers_to_remove].in_features

        # build the new model
        old_layers = list(pretrained.children())[:-layers_to_remove]
        if input_size != 3:
            first_conv = old_layers[0]
            old_layers[0] = nn.Conv2d(in_channels=input_size,
                                      out_channels=first_conv.out_channels,
                                      padding=first_conv.padding,
                                      kernel_size=first_conv.kernel_size,
                                      stride=first_conv.stride)
        self.new_model = nn.Sequential(*old_layers)
        self.fc = nn.Linear(in_features, num_features)

        # needed only if we want to do classification
        self.fc2 = nn.Linear(num_features, num_classes)

        self.train_only_fc = train_only_fc
        if self.train_only_fc:
            for params in self.new_model.parameters():
                params.requires_grad = False

    def forward(self, x):
        x = self.new_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        pred = self.fc2(x)
        return pred, x
