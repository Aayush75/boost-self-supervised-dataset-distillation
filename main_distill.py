import os
import yaml
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms
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
    print("="*80)
    print("SELF-SUPERVISED DATASET DISTILLATION")
    print("="*80)
    
    total_start_time = time.time()
    
    config_path = 'configs/cifar100.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device: {}".format(device))
    
    os.makedirs(config['saving']['distilled_assets_dir'], exist_ok=True)
    
    print("\n⏱️  Loading dataset and teacher model...")
    load_start = time.time()
    train_dataset, _ = get_dataset(config['data']['name'])
    teacher_model = get_teacher_model(config['models']['teacher']['path'], config['models']['teacher']['feature_dim']).to(device)
    load_time = time.time() - load_start
    print("✓ Loading completed in {:.2f}s".format(load_time))
    
    print("\n⏱️  Initializing distilled data parameters...")
    init_start = time.time()
    all_images_np = extract_all_data(train_dataset)
    all_images_torch = torch.from_numpy(all_images_np)
    
    print("  📊 Computing image PCA...")
    pca_img_start = time.time()
    downsampler = nn.Upsample(size=(config['parametrization']['image_basis_size'], config['parametrization']['image_basis_size']), mode='bilinear')
    all_images_down_flat = downsampler(all_images_torch).view(len(all_images_np), -1).numpy()
    pca_img = PCA(n_components=config['parametrization']['image_bases_U'], random_state=42).fit(all_images_down_flat)
    B_x_init = pca_img.components_
    sample_indices = np.random.choice(len(all_images_np), config['distillation']['num_distilled_images_m'], replace=False)
    C_x_init = pca_img.transform(all_images_down_flat[sample_indices])
    pca_img_time = time.time() - pca_img_start
    print("  ✓ Image PCA completed in {:.2f}s".format(pca_img_time))

    print("  📊 Computing representation PCA...")
    pca_repr_start = time.time()
    all_reprs_np = []
    with torch.no_grad():
        repr_loader = DataLoader(train_dataset, batch_size=512, shuffle=False, num_workers=4)
        for batch_idx, (images, _) in enumerate(tqdm(repr_loader, desc="Generating teacher representations")):
            batch_start = time.time()
            reprs = teacher_model(images.to(device)).cpu().numpy()
            all_reprs_np.append(reprs)
            if batch_idx == 0:
                print("    First batch processed in {:.3f}s".format(time.time() - batch_start))
    all_reprs_np = np.concatenate(all_reprs_np)
    pca_repr = PCA(n_components=config['parametrization']['repr_bases_V'], random_state=42).fit(all_reprs_np)
    B_y_init = pca_repr.components_
    C_y_init = pca_repr.transform(all_reprs_np[sample_indices])
    pca_repr_time = time.time() - pca_repr_start
    print("  ✓ Representation PCA completed in {:.2f}s".format(pca_repr_time))
        
    print("  📊 Initializing augmentation representations...")
    aug_init_start = time.time()
    C_aug_y_init = []
    sample_images_torch = all_images_torch[sample_indices].to(device)
    
    # Handle rotation augmentations
    with torch.no_grad():
        for rot_angle in tqdm(config['augmentations']['rotate'], desc="Projecting rotation representations"):
            aug_images = TF.rotate(sample_images_torch, angle=float(rot_angle))
            aug_reprs = teacher_model(aug_images).cpu().numpy()
            c_aug_y = pca_repr.transform(aug_reprs)
            C_aug_y_init.append(c_aug_y)
    
    # Handle color jitter augmentations if specified
    if 'color_jitter_strengths' in config['augmentations']:
        color_jitter = transforms.ColorJitter(**config['augmentations']['color_jitter'])
        with torch.no_grad():
            for strength in tqdm(config['augmentations']['color_jitter_strengths'], desc="Projecting color jitter representations"):
                aug_images = color_jitter(sample_images_torch)
                aug_reprs = teacher_model(aug_images).cpu().numpy()
                c_aug_y = pca_repr.transform(aug_reprs)
                C_aug_y_init.append(c_aug_y)
    
    # Handle crop augmentations if specified
    if 'crop_scales' in config['augmentations']:
        random_crop = transforms.RandomResizedCrop(**config['augmentations']['random_crop'])
        with torch.no_grad():
            for scale in tqdm(config['augmentations']['crop_scales'], desc="Projecting crop representations"):
                aug_images = random_crop(sample_images_torch)
                aug_reprs = teacher_model(aug_images).cpu().numpy()
                c_aug_y = pca_repr.transform(aug_reprs)
                C_aug_y_init.append(c_aug_y)
    
    # Handle gaussian blur augmentations if specified
    if 'gaussian_blur' in config['augmentations']:
        with torch.no_grad():
            for kernel_size in tqdm(config['augmentations']['gaussian_blur']['kernel_size'], desc="Projecting gaussian blur representations"):
                sigma = np.random.uniform(*config['augmentations']['gaussian_blur']['sigma'])
                aug_images = transforms.functional.gaussian_blur(sample_images_torch, kernel_size, sigma)
                aug_reprs = teacher_model(aug_images).cpu().numpy()
                c_aug_y = pca_repr.transform(aug_reprs)
                C_aug_y_init.append(c_aug_y)

    init_params = {"B_x": B_x_init, "B_y": B_y_init, "C_x": C_x_init, "C_y": C_y_init, "C_aug_y": C_aug_y_init}
    print(f"  ✓ Initialized {len(C_aug_y_init)} augmentation representations")
    distilled_data = DistilledData(init_params, config).to(device)
    
    init_total_time = time.time() - init_start
    print("✓ Initialization completed in {:.2f}s".format(init_total_time))
    
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
        
    # Add gradient clipping
    max_grad_norm = config.get('optimization', {}).get('gradient_clip', 1.0)
    
    print("\n⏱️  Starting distillation ({} steps)...".format(config['distillation']['steps']))
    distill_start = time.time()
        
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
        K_ts = f_w_X_t @ f_w_X_s.T
        pred_reprs_outer = K_ts @ krr_weights

        loss_outer = F.mse_loss(pred_reprs_outer, real_reprs_target)
        
        optimizer_distill.zero_grad()
        loss_outer.backward()
        
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(distilled_data.parameters(), max_grad_norm)
        
        optimizer_distill.step()
        scheduler.step()
        
        inner_model.train()
        optimizer_inner = optimizers_pool[pool_idx]

        # Collect all augmented images
        X_s_list_aug = [X_s_base.detach()]
        aug_counter = 0
        
        # Add rotation augmentations
        for rot_angle in config['augmentations']['rotate']:
            aug_images = TF.rotate(X_s_base.detach(), angle=float(rot_angle))
            X_s_list_aug.append(aug_images)
            aug_counter += 1
        
        # Add color jitter augmentations if specified
        if 'color_jitter_strengths' in config['augmentations']:
            color_jitter = transforms.ColorJitter(**config['augmentations']['color_jitter'])
            for _ in config['augmentations']['color_jitter_strengths']:
                aug_images = color_jitter(X_s_base.detach())
                X_s_list_aug.append(aug_images)
                aug_counter += 1
        
        # Add crop augmentations if specified
        if 'crop_scales' in config['augmentations']:
            random_crop = transforms.RandomResizedCrop(**config['augmentations']['random_crop'])
            for _ in config['augmentations']['crop_scales']:
                aug_images = random_crop(X_s_base.detach())
                X_s_list_aug.append(aug_images)
                aug_counter += 1
        
        # Add gaussian blur augmentations if specified
        if 'gaussian_blur' in config['augmentations']:
            for kernel_size in config['augmentations']['gaussian_blur']['kernel_size']:
                sigma = np.random.uniform(*config['augmentations']['gaussian_blur']['sigma'])
                aug_images = transforms.functional.gaussian_blur(X_s_base.detach(), kernel_size, sigma)
                X_s_list_aug.append(aug_images)
                aug_counter += 1
        
        Y_s_list_aug = distilled_data.reconstruct_representations()
        
        train_x = torch.cat(X_s_list_aug, dim=0)
        train_y = torch.cat([y.detach() for y in Y_s_list_aug], dim=0)

        # Debug: Print shapes to verify they match
        if step == 0:  # Only print on first step to avoid spam
            print(f"Debug - train_x shape: {train_x.shape}, train_y shape: {train_y.shape}")
            print(f"Debug - Number of image augmentations: {len(X_s_list_aug)}")
            print(f"Debug - Number of repr augmentations: {len(Y_s_list_aug)}")

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
    
    distill_total_time = time.time() - distill_start
    print("✓ Distillation completed in {:.1f}h".format(distill_total_time/3600))
        
    print("\n⏱️  Training Approximation Networks...")
    approx_start = time.time()
    distilled_data.eval()
    
    # Create approximation networks for all augmentations
    approx_networks = []
    aug_types = []
    
    # Rotation networks
    for rot_angle in config['augmentations']['rotate']:
        net = ApproximationMLP(
            num_repr_bases_V=config['parametrization']['repr_bases_V'],
            hidden_dim=config['models']['approximation_mlp']['hidden_dim'],
            num_layers=config['models']['approximation_mlp']['num_layers']
        ).to(device)
        approx_networks.append(net)
        aug_types.append('rot_{}'.format(rot_angle))
    
    # Color jitter networks
    if 'color_jitter_strengths' in config['augmentations']:
        for strength in config['augmentations']['color_jitter_strengths']:
            net = ApproximationMLP(
                num_repr_bases_V=config['parametrization']['repr_bases_V'],
                hidden_dim=config['models']['approximation_mlp']['hidden_dim'],
                num_layers=config['models']['approximation_mlp']['num_layers']
            ).to(device)
            approx_networks.append(net)
            aug_types.append('color_{}'.format(strength))
    
    # Crop networks
    if 'crop_scales' in config['augmentations']:
        for scale in config['augmentations']['crop_scales']:
            net = ApproximationMLP(
                num_repr_bases_V=config['parametrization']['repr_bases_V'],
                hidden_dim=config['models']['approximation_mlp']['hidden_dim'],
                num_layers=config['models']['approximation_mlp']['num_layers']
            ).to(device)
            approx_networks.append(net)
            aug_types.append('crop_{}'.format(scale))
    
    # Gaussian blur networks
    if 'gaussian_blur' in config['augmentations']:
        for kernel_size in config['augmentations']['gaussian_blur']['kernel_size']:
            net = ApproximationMLP(
                num_repr_bases_V=config['parametrization']['repr_bases_V'],
                hidden_dim=config['models']['approximation_mlp']['hidden_dim'],
                num_layers=config['models']['approximation_mlp']['num_layers']
            ).to(device)
            approx_networks.append(net)
            aug_types.append('blur_{}'.format(kernel_size))
    
    with torch.no_grad():
        C_y_base_target = distilled_data.C_y
        
    approx_times = []
    for i, (net, aug_type) in enumerate(zip(approx_networks, aug_types)):
        net_start = time.time()
        optimizer_approx = torch.optim.Adam(net.parameters(), lr=0.01)
        with torch.no_grad():
            C_y_aug_target = distilled_data.C_aug_y[i]
            target_shift = C_y_aug_target - C_y_base_target

        for epoch in tqdm(range(500), desc="Training Approximation Net for {}".format(aug_type), leave=False):
            optimizer_approx.zero_grad()
            pred_shift = net(C_y_base_target)
            loss = F.mse_loss(pred_shift, target_shift)
            loss.backward()
            optimizer_approx.step()
        
        net_time = time.time() - net_start
        approx_times.append(net_time)
        print("  ✓ {} network trained in {:.2f}s".format(aug_type, net_time))
    
    approx_total_time = time.time() - approx_start
    print("✓ All approximation networks trained in {:.2f}s".format(approx_total_time))

    print("\n⏱️  Saving distilled assets...")
    save_start = time.time()
    asset_dir = config['saving']['distilled_assets_dir']
    torch.save(distilled_data.state_dict(), os.path.join(asset_dir, 'distilled_data.pth'))
    
    for i, (net, aug_type) in enumerate(zip(approx_networks, aug_types)):
        torch.save(net.state_dict(), os.path.join(asset_dir, 'approx_net_{}.pth'.format(aug_type)))
    
    save_time = time.time() - save_start
    print("✓ Assets saved in {:.2f}s".format(save_time))
        
    total_time = time.time() - total_start_time
    
    print("\n{}".format("="*80))
    print("DISTILLATION SUMMARY")
    print("{}".format("="*80))
    print("Total Time: {:.1f} hours".format(total_time/3600))
    print("  Loading: {:.1f}s".format(load_time))
    print("  Initialization: {:.1f}s".format(init_total_time))
    print("  Distillation: {:.1f}h".format(distill_total_time/3600))
    print("  Approximation Networks: {:.1f}s".format(approx_total_time))
    print("  Saving: {:.1f}s".format(save_time))
    print("Saved {} approximation networks for: {}".format(len(approx_networks), aug_types))
    print("Assets saved to: {}".format(asset_dir))
    print("{}".format("="*80))

if __name__ == "__main__":
    distill()