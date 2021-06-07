# import torchvision.models # for torchvision models

import models

modelname_to_func = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    # 'vgg11': models.vgg11,
    # 'vgg13': models.vgg13,
}
