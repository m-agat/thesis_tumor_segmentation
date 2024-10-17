import models.models as models
import os 
import torch
import config.config as config 
from monai.inferers import sliding_window_inference
from functools import partial
import numpy as np 

model= models.swinunetr_model

model.load_state_dict(torch.load(os.path.join(config.root_dir, "results/swinunetr_model.pt"))["state_dict"])
model.to(config.device)
model.eval()

model_inferer_test = partial(
    sliding_window_inference,
    roi_size=[config.roi[0], config.roi[1], config.roi[2]],
    sw_batch_size=1,
    predictor=model,
    overlap=0.6,
)

with torch.no_grad():
    for batch_data in config.test_loader:
        image = batch_data["image"].cuda()
        prob = torch.sigmoid(model_inferer_test(image))
        seg = prob[0].detach().cpu().numpy()
        seg = (seg > 0.5).astype(np.int8)
        print(seg.shape)
        seg_out = np.zeros((seg.shape[1], seg.shape[2]))
        seg_out[seg[1] == 1] = 2
        seg_out[seg[0] == 1] = 1
        seg_out[seg[2] == 1] = 4