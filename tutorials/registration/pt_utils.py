import torch


## transformation functions
def get_reference_grid(grid_size):
    # grid_size: [batch_size, height, width]
    grid = torch.cast(torch.stack(torch.meshgrid(
                        torch.range(grid_size[1]),
                        torch.range(grid_size[2])), axis=2), dtype=tf.float32)
    return tf.tile(tf.expand_dims(grid, axis=0), [grid_size[0],1,1,1])

def warp_images(images, ddfs):
    # images: [batch_size, height, width]
    # ddfs: [batch_size, height, width, 2]
    reference_grid = get_reference_grid(ddfs.shape[0:3])
    warped_grids = reference_grid + ddfs
    return grid_sample(images, warped_grids)

def get_reference_grid3d(img):
    shape = img.shape
    if len(img.shape) != 3:
        shape = shape[-3:]
    mesh_points = [torch.linspace(-1, 1, dim) for dim in shape]
    grid = torch.stack(torch.meshgrid(*mesh_points))  # shape:[3, x, y, z]
    grid = torch.stack([grid]*img.shape[0])  # add batch
    grid = grid.type(torch.FloatTensor)
    return grid.cuda()


def warp3d(img, ddf):
    assert img.shape[-3:] == ddf.shape[-3:], "Shapes not consistent btw img and ddf."
    grid = get_reference_grid3d(img)
    new_grid = grid + ddf  # [batch, 3, x, y, z]
    new_grid = new_grid.permute(0, 2, 3, 4, 1)
    new_grid = new_grid[..., [2, 1, 0]]
    return F.grid_sample(img, new_grid, mode='bilinear', align_corners=False)
    
## loss functions
def square_difference(i1, i2):
    return torch.nn.functional.mse_loss(y_pred, y_true)


def gradient_dx(fv):
    return (fv[:, 2:, 1:-1] - fv[:, :-2, 1:-1]) / 2


def gradient_dy(fv):
    return (fv[:, 1:-1, 2:] - fv[:, 1:-1, :-2]) / 2


def gradient_txy(txy, fn):
    return torch.stack([fn(txy[..., i]) for i in [0, 1]], axis=3)


def gradient_norm(displacement, flag_l1=False):
    dtdx = gradient_txy(displacement, gradient_dx)
    dtdy = gradient_txy(displacement, gradient_dy)
    if flag_l1:
        norms = torch.abs(dtdx) + torch.abs(dtdy)
    else:
        norms = dtdx**2 + dtdy**2
    return torch.reduce_mean(norms, [1, 2, 3])