"""
turtorial code for torch lighting 
"""

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
imgs_dist1 = lambda: torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
imgs_dist2 = lambda: torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)
metric = FrechetInceptionDistance(feature=64)
values = [ ]
for _ in range(3):
    metric.update(imgs_dist1(), real=True)
    metric.update(imgs_dist2(), real=False)
    values.append(metric.compute()) 
    metric.reset()
fig_, ax_ = metric.plot(values)

# Print the values
print(values)