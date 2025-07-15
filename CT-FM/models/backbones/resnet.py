import monai
import torch


class ResNet(monai.networks.nets.ResNet):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # ---- If needed avg pool should be done by the SSL framework ----
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # if self.fc is not None:
        #     x = self.fc(x)
        # ----------------------------------------------------------------

        return x
