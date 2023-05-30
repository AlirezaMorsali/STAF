#!/usr/bin/env jupyter

fig, axes = plt.subplots(1, figsize=(6, 6))
# axes.imshow(model_output.cpu().view(256, 256).detach().numpy())
axes.imshow(img.view(256, 256).detach().numpy())


