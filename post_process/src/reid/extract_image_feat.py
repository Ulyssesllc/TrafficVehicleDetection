"""Extract image feature for both det/mot image feature."""

import os
import sys
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from reid.reid_inference.reid_model import build_reid_model

sys.path.append("../")


class ReidFeature:
    """Extract reid feature."""

    def __init__(self, model_name, gpu_id):
        """
        Available models:
        - "resnext101_ibn_a"
        - "resnet101_ibn_a"
        """
        print("init reid model")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.model, self.reid_cfg = build_reid_model(model_name)
        device = torch.device("cuda")
        self.model = self.model.to(device)
        self.model.eval()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.val_transforms = T.Compose(
            [
                T.Resize(self.reid_cfg.INPUT.SIZE_TEST, interpolation=3),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def extract(self, img_list, batch_size=64):
        """
        Extract image features in batches to avoid CUDA memory overflow.
        Args:
            img_list (list): List of images to process.
            batch_size (int): Maximum number of images to process at once.
        Returns:
            np.ndarray: Features for all images, shape (N, 2048), where N is the total number of images.
        """
        all_features = []
        num_images = len(img_list)

        for start_idx in range(0, num_images, batch_size):
            # Get the batch of images
            end_idx = min(start_idx + batch_size, num_images)
            batch_imgs = img_list[start_idx:end_idx]

            # Prepare the batch
            img_batch = []
            for img in batch_imgs:
                img = self.val_transforms(img)
                img = img.unsqueeze(0)
                img_batch.append(img)
            img_tensor = torch.cat(img_batch, dim=0)

            with torch.no_grad():
                img_tensor = img_tensor.to("cuda")
                flip_feats = self.reid_cfg.TEST.FLIP_FEATS == "yes"

                if flip_feats:
                    for i in range(2):
                        if i == 1:
                            inv_idx = torch.arange(img_tensor.size(3) - 1, -1, -1).long().cuda()
                            img_tensor = img_tensor.index_select(3, inv_idx)
                            feat1 = self.model(img_tensor)
                        else:
                            feat2 = self.model(img_tensor)
                    feat = feat2 + feat1
                else:
                    feat = self.model(img_tensor)

            # Append features to the result list
            all_features.append(feat.cpu().detach().numpy())

        # Concatenate all features along the batch dimension
        all_features = np.concatenate(all_features, axis=0)
        return all_features


    # CUDA of out memory risk
    def extract_free_batch(self, img_list):
        """Extract image feature with given image path.
        Feature shape (2048,) float32."""

        img_batch = []
        for img in img_list:
            # img = Image.open(img_path).convert('RGB')
            img = self.val_transforms(img)
            img = img.unsqueeze(0)
            img_batch.append(img)
        img = torch.cat(img_batch, dim=0)

        with torch.no_grad():
            img = img.to("cuda")
            flip_feats = False
            if self.reid_cfg.TEST.FLIP_FEATS == "yes":
                flip_feats = True
            if flip_feats:
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
                        img = img.index_select(3, inv_idx)
                        feat1 = self.model(img)
                    else:
                        feat2 = self.model(img)
                feat = feat2 + feat1
            else:
                feat = self.model(img)
        feat = feat.cpu().detach().numpy()
        return feat


def debug_reid_feat():
    """Debug reid feature to make sure the same with Track2."""

    exp_reidfea = ReidFeature(0)
    feat = exp_reidfea.extract(
        ["crop_test/00001.jpg", "crop_test/00002.jpg", "crop_test/00003.jpg"]
    )
    print(feat)


def main():
    """Main method."""
    debug_reid_feat()



if __name__ == "__main__":
    main()
