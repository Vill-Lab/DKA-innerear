import torch


def SuPreM_loader(model, ckpt_path, decoder=True, encoder_only=False, device=torch.device("cuda:0")):
    model_dict = torch.load(ckpt_path, map_location=device)["net"]
    store_dict = model.state_dict()
    amount = 0
    if model.__class__.__name__ == "SegResNet":
        print("Loading SuPreM SegResNet backbone pretrained weights")
        decoder_keys = ["up_layers", "up_samples", "conv_final"]
        for key in model_dict.keys():
            new_key = ".".join(key.split(".")[1:])
            if new_key in store_dict.keys() and "conv_final.2.conv" not in new_key:
                if not decoder and any(decoder_key in new_key for decoder_key in decoder_keys):
                    continue
                store_dict[new_key] = model_dict[key]
                amount += 1

    elif model.__class__.__name__ == "UNet3D":
        print("Loading SuPreM UNet backbone pretrained weights")
        decoder_keys = ["up_tr", "out_tr"]
        for key in model_dict.keys():
            new_key = ".".join(key.split(".")[2:])
            if new_key in store_dict.keys():
                if not decoder and any(decoder_key in new_key for decoder_key in decoder_keys):
                    continue
                store_dict[new_key] = model_dict[key]
                amount += 1
    else:
        raise ValueError("Model not supported")

    model.load_state_dict(store_dict)

    if encoder_only:

        class Encoder(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                if self.model.__class__.__name__ == "SegResNet":
                    self.out = self.model.encode(x)[0]
                    return self.out

                elif self.model.__class__.__name__ == "UNet3D":
                    self.out64, self.skip_out64 = self.model.down_tr64(x)
                    self.out128, self.skip_out128 = self.model.down_tr128(self.out64)
                    self.out256, self.skip_out256 = self.model.down_tr256(self.out128)
                    self.out512 = self.model.down_tr512(self.out256)[0]
                    return self.out512
                else:
                    raise ValueError("Model not supported")

        model = Encoder(model)

    print(f"Loaded {amount}/{len(store_dict.keys())} keys")
    return model
