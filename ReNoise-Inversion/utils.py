import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# def resize_image(image, size):
#     '''
#     Input
#     -----
#     image : torch.Tensor (C, H, W)
#     Return
#     ------
#     image: torch.Tensor
#     h : original height, 'original_size' of SDXL
#     w : original width, 'original_size' of SDXL
#     c_top : top corner of the crop, 'crops_coords_top_left' of SDXL
#     c_left : left corner of the crop, 'crops_coords_top_left' of SDXL
#     '''
#     # Original image size
#     img_tensor = transforms.PILToTensor()(image)
#     c, h, w = img_tensor.shape
#     device = 'cpu'

#     # Resize w.r.t the smaller dimension
#     if h > w:
#         new_w = size
#         new_h = int(h * size / w)
#         c_top = int(np.random.uniform(0, new_h - size))
#         c_left = 0
#     else:
#         new_h = size
#         new_w = int(w * size / h)
#         c_top = 0
#         c_left = int(np.random.uniform(0, new_w - size))

#     # Resize the image
#     # image = torch.nn.functional.interpolate(image.unsqueeze(0).cpu(), (new_h, new_w), mode='bilinear', align_corners=False)
#     # image = transforms.ToPILImage()(image.cpu())
#     # image = transforms.Resize((new_h, new_w))(image)
#     print(f"new_h : {new_h}, new_w : {new_w}")
#     image = image.resize((new_h,new_w), Image.NEAREST)
#     image = image.crop((c_left, c_top, c_left + size, c_top + size))
#     # image = transforms.PILToTensor()(image).to(device).unsqueeze(0)

#     # # Round
#     # image = torch.round(image)
#     # image = torch.clamp(image, 0, 255)
#     # Crop with top-left corner
#     # image = image[:, :, c_top:c_top + size, c_left:c_left + size].clone()


#     return image, h, w, c_top, c_left

def resize_image(image, size):
    '''
    Input
    -----
    image : torch.Tensor (C, H, W)
    Return
    ------
    image: torch.Tensor
    h : original height
    w : original width
    c_top : top corner of the crop
    c_left : left corner of the crop
    '''
    # Original image size
    c, h, w = image.shape

    # Resize w.r.t the smaller dimension
    if h > w:
        new_w = size
        new_h = int(h * size / w)
        c_top = int(np.random.uniform(0, new_h - size))
        c_left = 0
    else:
        new_h = size
        new_w = int(w * size / h)
        c_top = 0
        c_left = int(np.random.uniform(0, new_w - size))

    # Resize the image
    print(f"new_h : {new_h}, new_w : {new_w}")
    image = torch.nn.functional.interpolate(image.unsqueeze(0), (new_h, new_w), mode='bilinear', align_corners=False)

    # Crop with top-left corner
    image = image[:, :, c_top:c_top + size, c_left:c_left + size]

    return image, h, w, c_top, c_left


class my_collate_fn(object):
    def __init__(self):
        pass
    def __call__(self, batch):
        with torch.no_grad():
            imgs, txts = zip(*batch)

            # Convert from tuple to list
            imgs = list(imgs)
            txts = list(txts)

            # Preprocess
            imgs_ = []
            hw_list = []
            c_top_left_list = []
            for img in imgs:
                img, h, w, c_top, c_left = resize_image(img, 1024)
                img = img.squeeze().cpu().numpy()
                img = np.transpose(img, (1, 2, 0))
                img = Image.fromarray(img)
                imgs_.append(img)
                hw_list.append((h, w))
                c_top_left_list.append((c_top, c_left))
            imgs_pil = imgs_

            txts = [txt[0] for txt in txts] # Select only the first text as the prompt

            return imgs_pil, txts, hw_list, c_top_left_list

