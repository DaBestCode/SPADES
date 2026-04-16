import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms.functional as TF
import h5py
import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm

# -------------------------------------------------------------------
# 1. THE FINAL CLEAN INFERENCE FILTERS (Synced with V26/Ensemble)
# -------------------------------------------------------------------

def step_remove_background_clutter_strong(img_tensor):
    """Aggressively zero-outs stars using multi-scale density stencil."""
    combined = torch.max(img_tensor, dim=0, keepdim=True)[0]
    dense_3x3 = F.avg_pool2d(combined.unsqueeze(0), 3, 1, 1).squeeze(0)
    dense_5x5 = F.avg_pool2d(combined.unsqueeze(0), 5, 1, 2).squeeze(0)
    mask = ((dense_3x3 > 0.25) & (dense_5x5 > 0.10)).float()
    safe_mask = F.max_pool2d(mask.unsqueeze(0), 3, 1, 1).squeeze(0)
    return img_tensor * safe_mask, safe_mask

def step_conditional_blur_v2(img_tensor, safe_mask, threshold_pct=0.60):
    """Softens high-res details if the satellite fills more than 60% of frame."""
    coverage = safe_mask.mean().item()
    if coverage > threshold_pct:
        img_tensor = TF.gaussian_blur(img_tensor, [5, 5], [1.0, 1.0])
    return img_tensor

def step_blue_floor_test(img_tensor):
    """Mild batch-norm alignment floor."""
    blue_bias = torch.zeros_like(img_tensor)
    blue_bias[2, :, :] = 0.05 
    blue_bias[1, :, :] = 0.01 
    return torch.max(img_tensor, blue_bias)

# -------------------------------------------------------------------
# 2. UPGRADED MODEL ARCHITECTURE: V25 (EffNet & ConvNeXt)
# -------------------------------------------------------------------

class SparkFinalNet(nn.Module):
    def __init__(self):
        super().__init__()
        # TRANSLATION: EfficientNet-V2-S
        effnet = models.efficientnet_v2_s(weights=None)
        self.backbone_t = effnet.features
        self.t_head = nn.Sequential(
            nn.Linear(1280 + 1, 512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, 3)
        )
        # ROTATION: ConvNeXt Tiny
        convnext = models.convnext_tiny(weights=None)
        self.backbone_r = convnext.features
        self.r_head = nn.Sequential(
            nn.Linear(768, 512), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(512, 4)
        )

    def forward(self, x_f, x_r, scale):
        f_t = torch.flatten(nn.AdaptiveAvgPool2d(1)(self.backbone_t(x_f)), 1)
        t = self.t_head(torch.cat([f_t, scale.view(-1, 1).to(f_t.dtype)], dim=1))
        f_r = torch.flatten(nn.AdaptiveAvgPool2d(1)(self.backbone_r(x_r)), 1)
        q_raw = self.r_head(f_r)
        q = q_raw / (q_raw.norm(dim=1, keepdim=True) + 1e-8)
        return t, q

# -------------------------------------------------------------------
# 3. GHOST-LIGHT TENSOR ENGINE
# -------------------------------------------------------------------
def create_v25_test_tensor(xs, ys, ts, ps, target_ts, window, off_weight, crop_box=None):
    img = np.zeros((3, 224, 224), dtype=np.float32)
    if len(xs) < 5: return torch.from_numpy(img).float(), 1.0
    weights = np.clip(1.0 - (target_ts - ts) / window, 0.2, 1.0)
    weights[ps == 0] *= off_weight 

    if crop_box:
        cx, cy, sz = crop_box
        sz, scale_hint = max(sz, 10.0), sz / 1280.0
        xi = np.clip((xs - (cx - sz/2)) * (224/sz), 0, 223).astype(int)
        yi = np.clip((ys - (cy - sz/2)) * (224/sz), 0, 223).astype(int)
    else:
        scale_hint, xi, yi = 1.0, np.clip(xs*(223/1280), 0, 223).astype(int), np.clip(ys*(223/720), 0, 223).astype(int)
    
    t_step = (ts[-1] - ts[0]) / 3.0
    for i in range(3):
        m = (ts >= ts[0] + i*t_step) & (ts < ts[0] + (i+1)*t_step)
        if m.any(): np.add.at(img[i], (yi[m], xi[m]), weights[m])
    
    tensor = torch.from_numpy(img).unsqueeze(0)
    dilated = F.pad(F.max_pool2d(tensor, 2, 1, 0), (0, 1, 0, 1))
    blur_k = torch.ones((1, 1, 3, 3)) / 9.0
    for c in range(3):
        blurred = F.conv2d(dilated[:, c:c+1], blur_k, padding=1)
        dilated[:, c:c+1] = dilated[:, c:c+1] + 0.8 * (dilated[:, c:c+1] - blurred)
    
    sh = dilated.squeeze(0).numpy()
    log_img = np.log1p(np.clip(sh, 0, None))
    sigmoid_img = 1 / (1 + np.exp(-10 * (log_img / (np.max(log_img) + 1e-9) - 0.3)))
    final_img = np.clip(sigmoid_img / (np.percentile(sigmoid_img, 99.5) + 1e-8), 0, 1)
    return torch.from_numpy(final_img).float(), float(scale_hint)

# -------------------------------------------------------------------
# 4. SUBMISSION ENGINE
# -------------------------------------------------------------------
def run_v25_submission():
    # PATH - Update to your best robust epoch
    MODEL_PATH = 'checkpoints_v25_robust_scratch/spark_v23_robust_e14.pth' 
    DATA_DIR = 'test_data/'
    SAVE_PATH = 'submission_v25_robust_FinalInference.csv'
    WINDOW_TICKS = 400 * 1000 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SparkFinalNet().to(device)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"🎯 V25 ConvNeXt Loaded: {MODEL_PATH}")
    else:
        print(f"⚠️ ERROR: {MODEL_PATH} not found."); return

    model.eval()
    df = pd.read_csv('template.csv')
    col_name = 'timestamp' if 'timestamp' in df.columns else 'filename'
    df['traj_id'] = df[col_name].apply(lambda x: x.split('_')[0])

    for traj_id, group in tqdm(df.groupby('traj_id'), desc="Trajectories"):
        h5_file = os.path.join(DATA_DIR, f"{traj_id}.h5")
        if not os.path.exists(h5_file): continue
            
        with h5py.File(h5_file, 'r') as f:
            ev_ts, ev_xs, ev_ys, ev_ps = f['events/ts'][:], f['events/xs'][:], f['events/ys'][:], f['events/ps'][:]
            on_ratio = np.mean(ev_ps[:500000]) if len(ev_ps) > 0 else 0.5
            off_weight = on_ratio / (1 - on_ratio + 1e-6)
            h_vis, _, _ = np.histogram2d(ev_ys[:1000000], ev_xs[:1000000], bins=[720, 1280], range=[[0, 720], [0, 1280]])
            hot_mask = h_vis < (np.mean(h_vis) + 15 * np.std(h_vis))

        for i in group.index:
            try:
                target_ts = int(df.loc[i, col_name].split('_')[1]) * 100000 
                idx_e, idx_s = np.searchsorted(ev_ts, target_ts), np.searchsorted(ev_ts, target_ts - WINDOW_TICKS)
                xs, ys, ps, ts = ev_xs[idx_s:idx_e], ev_ys[idx_s:idx_e], ev_ps[idx_s:idx_e], ev_ts[idx_s:idx_e]

                valid = hot_mask[np.clip(ys.astype(int), 0, 719), np.clip(xs.astype(int), 0, 1279)]
                xs, ys, ps, ts = xs[valid], ys[valid], ps[valid], ts[valid]

                if len(xs) < 15:
                    df.iloc[i, 1:8] = [0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 1.0]; continue

                # Global Pass (img_f)
                img_f, _ = create_v25_test_tensor(xs, ys, ts, ps, target_ts, WINDOW_TICKS, off_weight)
                img_f_in = step_blue_floor_test(img_f.unsqueeze(0)).to(device)

                # Local Zoom Pass (img_r)
                cx, cy = np.median(xs), np.median(ys)
                std_crop = np.clip(max(np.std(xs), np.std(ys)) * 12.0, 300, 1000)
                img_r, sc = create_v25_test_tensor(xs, ys, ts, ps, target_ts, WINDOW_TICKS, off_weight, (cx, cy, std_crop))
                
                # --- APPLY CLEAN INFERENCE PIPELINE ---
                img_r_in = img_r.unsqueeze(0)
                img_r_in, s_mask = step_remove_background_clutter_strong(img_r_in)
                img_r_in = step_conditional_blur_v2(img_r_in, s_mask, 0.60)
                img_r_in = step_blue_floor_test(img_r_in).to(device)
                
                sc_final = torch.tensor([[sc]]).to(device)

                with torch.no_grad(), torch.amp.autocast('cuda'):
                    t_fin, q_fin = model(img_f_in, img_r_in, sc_final)
                
                df.iloc[i, 1:4] = t_fin.cpu().squeeze().numpy()
                df.iloc[i, 4:8] = q_fin.cpu().squeeze().numpy()

            except Exception:
                df.iloc[i, 1:8] = [0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 1.0]

    df.drop(columns=['traj_id']).to_csv(SAVE_PATH, index=False)
    print(f"✅ V25 Submission complete: {SAVE_PATH}")

if __name__ == "__main__":
    run_v25_submission()