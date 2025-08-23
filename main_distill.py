import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms import functional as TF
from tqdm import tqdm
from sklearn.decomposition import PCA

from utils import get_dataset, extract_all_data
from models import InnerCNN, ApproximationMLP, get_teacher_model

class DistilledData(nn.Module):
    def __init__(self, init_params, config):
        super().__init__()
        self.config = config
        self.upsampler = nn.Upsample(
            size=(config['data']['resolution'][1], config['data']['resolution'][2]),
            mode='bilinear',
            align_corners=False
        )
        
        self.B_x = nn.Parameter(torch.tensor(init_params['B_x'], dtype=torch.float32))
        self.B_y = nn.Parameter(torch.tensor(init_params['B_y'], dtype=torch.float32))
        self.C_x = nn.Parameter(torch.tensor(init_params['C_x'], dtype=torch.float32))
        self.C_y = nn.Parameter(torch.tensor(init_params['C_y'], dtype=torch.float32))

        self.C_aug_y = nn.ParameterList([
            nn.Parameter(torch.tensor(p, dtype=torch.float32)) for p in init_params['C_aug_y']
        ])
        
    def reconstruct_images(self):
        images_small = self.C_x @ self.B_x
        num_images = self.config['distillation']['num_distilled_images_m']
        ch = self.config['data']['resolution'][0]
        sz = self.config['parametrization']['image_basis_size']
        images_small = images_small.view(num_images, ch, sz, sz)
        return self.upsampler(images_small)

    def reconstruct_representations(self):
        base_repr = self.C_y @ self.B_y
        aug_reprs = [c_aug @ self.B_y for c_aug in self.C_aug_y]
        return [base_repr] + aug_reprs
    
def distill():
    config_path = 'configs/cifar100.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    os.makedirs(config['saving']['distilled_assets_dir'], exist_ok=True)
    
    train_dataset, _ = get_dataset(config['data']['name'])
    teacher_model = get_teacher_model(config['models']['teacher']['path'], config['models']['teacher']['feature_dim']).to(device)
    
    print("Initializing distilled data parameters")
    all_images_np = extract_all_data(train_dataset)
    all_images_torch = torch.from_numpy(all_images_np)
    
    downsampler = nn.Upsample(size=(config['parametrization']['image_basis_size'], config['parametrization']['image_basis_size']), mode='bilinear')
    all_images_down_flat = downsampler(all_images_torch).view(len(all_images_np), -1).numpy()
    pca_img = PCA(n_components=config['parametrization']['image_bases_U'], random_state=42).fit(all_images_down_flat)
    B_x_init = pca_img.components_
    sample_indices = np.random.choice(len(all_images_np), config['distillation']['num_distilled_images_m'], replace=False)
    C_x_init = pca_img.transform(all_images_down_flat[sample_indices])

    all_reprs_np = []
    with torch.no_grad():
        repr_loader = DataLoader(train_dataset, batch_size=512, shuffle=False, num_workers=4)
        for images, _ in tqdm(repr_loader, desc="Generating all teacher representations for PCA"):
            all_reprs_np.append(teacher_model(images.to(device)).cpu().numpy())
    all_reprs_np = np.concatenate(all_reprs_np)
    pca_repr = PCA(n_components=config['parametrization']['repr_bases_V'], random_state=42).fit(all_reprs_np)
    B_y_init = pca_repr.components_
    C_y_init = pca_repr.transform(all_reprs_np[sample_indices])
        
    print("Initializing C_aug_y with augmented teacher representations")
    C_aug_y_init = []
    sample_images_torch = all_images_torch[sample_indices].to(device)
    with torch.no_grad():
        for rot_angle in tqdm(config['augmentations']['rotate'], desc="Projecting augmentation representations"):
            aug_images = TF.rotate(sample_images_torch, angle=float(rot_angle))
            aug_reprs = teacher_model(aug_images).cpu().numpy()
            c_aug_y = pca_repr.transform(aug_reprs)
            C_aug_y_init.append(c_aug_y)

    init_params = {"B_x": B_x_init, "B_y": B_y_init, "C_x": C_x_init, "C_y": C_y_init, "C_aug_y": C_aug_y_init}
    distilled_data = DistilledData(init_params, config).to(device)
    
    optimizer_distill = torch.optim.AdamW(distilled_data.parameters(), lr=config['distillation']['optimizer']['lr'])
    lr_schedule = lambda step: 1.0 - step / config['distillation']['steps']
    scheduler = LambdaLR(optimizer_distill, lr_lambda=lr_schedule)
    
    model_pool = [InnerCNN(feature_dim=config['models']['inner_cnn']['feature_dim']).to(device) for _ in range(config['model_pool']['size_L'])]
    optimizers_pool = [torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9) for model in model_pool]
    step_counters = [0] * config['model_pool']['size_L']
    outer_loss_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4)
    real_images_outer = next(iter(outer_loss_loader))[0].to(device)
    with torch.no_grad():
        real_reprs_target = teacher_model(real_images_outer)
        
    for step in tqdm(range(config['distillation']['steps']), desc="Distilling Dataset"):
        distilled_data.train()
        pool_idx = np.random.randint(config['model_pool']['size_L'])
        inner_model = model_pool[pool_idx]
        inner_model.eval()
        
        X_s_base = distilled_data.reconstruct_images()
        Y_s_base = distilled_data.reconstruct_representations()[0]
        
        with torch.no_grad():
            f_w_X_t = inner_model(real_images_outer)
        f_w_X_s = inner_model(X_s_base)
        
        K_ss = f_w_X_s @ f_w_X_s.T
        krr_weights = torch.linalg.solve(K_ss + 1e-4 * torch.eye(K_ss.size(0), device=device), Y_s_base)
        pred_reprs_outer = f_w_X_t @ krr_weights
        loss_outer = F.mse_loss(pred_reprs_outer, real_reprs_target)
        
        optimizer_distill.zero_grad()
        loss_outer.backward()
        optimizer_distill.step()
        scheduler.step()
        
        inner_model.train()
        optimizer_inner = optimizers_pool[pool_idx]

        X_s_list_aug = [X_s_base]
        for rot_angle in config['augmentations']['rotate']:
            aug_images = TF.rotate(X_s_base, angle=float(rot_angle))
            X_s_list_aug.append(aug_images)
        
        Y_s_list_aug = distilled_data.reconstruct_representations()
        
        train_x = torch.cat(X_s_list_aug, dim=0)
        train_y = torch.cat(Y_s_list_aug, dim=0)

        pred_y = inner_model(train_x)
        loss_inner = F.mse_loss(pred_y, train_y)
        
        optimizer_inner.zero_grad()
        loss_inner.backward()
        optimizer_inner.step()

        step_counters[pool_idx] += 1
        if step_counters[pool_idx] >= config['model_pool']['inner_loop_steps_Z']:
            model_pool[pool_idx] = InnerCNN(feature_dim=config['models']['inner_cnn']['feature_dim']).to(device)
            optimizers_pool[pool_idx] = torch.optim.SGD(model_pool[pool_idx].parameters(), lr=0.1, momentum=0.9)
            step_counters[pool_idx] = 0
        
    print("\nTraining Approximation Networks")
    distilled_data.eval()
    approx_networks = [
        ApproximationMLP(
            num_repr_bases_V=config['parametrization']['repr_bases_V'],
            hidden_dim=config['models']['approximation_mlp']['hidden_dim']
        ).to(device) for _ in config['augmentations']['rotate']
    ]
    
    with torch.no_grad():
        C_y_base_target = distilled_data.C_y
        
    for i, rot_angle in enumerate(config['augmentations']['rotate']):
        net = approx_networks[i]
        optimizer_approx = torch.optim.Adam(net.parameters(), lr=0.01)
        with torch.no_grad():
            C_y_aug_target = distilled_data.C_aug_y[i]
            target_shift = C_y_aug_target - C_y_base_target

        for epoch in tqdm(range(500), desc=f"Training Approximation Net for Rot {rot_angle} degrees"):
            optimizer_approx.zero_grad()
            pred_shift = net(C_y_base_target)
            loss = F.mse_loss(pred_shift, target_shift)
            loss.backward()
            optimizer_approx.step()

    print("Saving Distilled Assets")
    asset_dir = config['saving']['distilled_assets_dir']
    torch.save(distilled_data.state_dict(), os.path.join(asset_dir, 'distilled_data.pth'))
    
    for i, net in enumerate(approx_networks):
        rot_angle = config['augmentations']['rotate'][i]
        torch.save(net.state_dict(), os.path.join(asset_dir, f'approx_net_rot_{rot_angle}.pth'))
        
    print(f"Distillation complete. Assets saved to: {asset_dir}")

if __name__ == "__main__":
    distill()