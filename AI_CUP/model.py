import torch
import torch.nn as nn

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[9,19,39], bottleneck_channels=32):
        super().__init__()
        self.use_bottleneck = bottleneck_channels > 0 and in_channels > 1

        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            conv_in_channels = bottleneck_channels
        else:
            conv_in_channels = in_channels

        self.conv_layers = nn.ModuleList([
            nn.Conv1d(conv_in_channels, out_channels, kernel_size=k, padding=k//2, bias=False)
            for k in kernel_sizes
        ])

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        )

        self.batch_norm = nn.BatchNorm1d(out_channels * (len(kernel_sizes) + 1))
        self.activation = nn.ReLU()

    def forward(self, x):
        input_tensor = x
        if self.use_bottleneck:
            x = self.bottleneck(x)

        conv_outputs = [conv(x) for conv in self.conv_layers]
        conv_outputs.append(self.maxpool_conv(input_tensor))
        x = torch.cat(conv_outputs, dim=1)
        x = self.batch_norm(x)
        return self.activation(x)

class InceptionTimeClassifier(nn.Module):
    def __init__(self, num_classes, in_channels=6, num_blocks=6, out_channels=32):
        super().__init__()
        self.blocks = nn.Sequential(*[
            InceptionBlock(
                in_channels if i == 0 else out_channels * 4,
                out_channels
            ) for i in range(num_blocks)
        ])
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(out_channels * 4, num_classes)

    def forward(self, x):
        x = self.blocks(x)                      # (batch, C, T)
        x = self.global_avg_pool(x).squeeze(-1) # (batch, C)
        return self.fc(x)                       # (batch, num_classes)

