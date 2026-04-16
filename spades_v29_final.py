import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.models as models
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import h5py
import numpy as np
import os, glob, random, math, io
import torch.nn.functional as F
from PIL import Image

# ==============================================================================
# 1. AUGMENTATION SUITE — V29 TORCH-NATIVE, ALL BUGS FIXED
#
#  Every function is self-contained, GPU-compatible (device-aware),
#  and annotated with what it fixes vs previous broken versions.
# ==============================================================================

def step_blue_floor(img_tensor):
    """
    ALWAYS-ON domain fix.
    Real images never truly black — facility + camera produce a permanent
    ambient floor: ~+15-25/255 blue, ~+5-10/255 green, ~+0-5/255 red.
    torch.max = floor semantics: raises dark pixels, never lowers bright ones.
    """
    bias = torch.zeros_like(img_tensor)
    bias[0] = random.uniform(0.00, 0.02)   # R floor
    bias[1] = random.uniform(0.01, 0.04)   # G floor
    bias[2] = random.uniform(0.06, 0.10)   # B floor (dominant)
    return torch.max(img_tensor, bias)


def step_structural_debris(img_tensor):
    """
    RANDOM (20% prob).
    Thin bright bars — satellite structural members, panel edges, mounting rig.
    """
    c, h, w = img_tensor.shape
    mask = torch.zeros((1, h, w), device=img_tensor.device)
    for _ in range(random.randint(1, 3)):
        cw, ch = random.randint(1, 3), random.randint(40, 120)
        if random.random() > 0.5: cw, ch = ch, cw
        cx = random.randint(0, max(0, w - cw))
        cy = random.randint(0, max(0, h - ch))
        beam = torch.rand(1, ch, cw, device=img_tensor.device) * 0.3 + 0.7
        mask[:, cy:cy+ch, cx:cx+cw] = torch.maximum(mask[:, cy:cy+ch, cx:cx+cw], beam)
    return torch.clamp(img_tensor + mask.repeat(3,1,1) * random.uniform(0.3, 0.8), 0.0, 1.0)


def step_edge_bright(img_tensor):
    """
    ALWAYS-ON domain fix.
    Satellite geometry edges appear blown-white in real data due to the
    directional studio light. Simulated via gaussian unsharp mask with a
    bright_mask guard — only boosts pixels already above 0.35 brightness.

    FIX vs V26/V29: old version used max_pool dilation diff with no guard,
    which added fake edge artifacts to the near-black background.
    """
    k = torch.tensor([[1,2,1],[2,4,2],[1,2,1]],
                      dtype=torch.float32, device=img_tensor.device) / 16.0
    k = k.view(1,1,3,3).repeat(3,1,1,1)
    padded  = F.pad(img_tensor.unsqueeze(0), (1,1,1,1), mode='reflect')
    blurred = F.conv2d(padded, k, groups=3).squeeze(0)
    edges   = img_tensor - blurred
    bright_mask = (img_tensor.max(0, keepdim=True)[0] > 0.35).float()
    boost = random.uniform(0.4, 0.7)
    return torch.clamp(img_tensor + edges * bright_mask * boost, 0.0, 1.0)


def step_organic_noise(img_tensor):
    """
    ALWAYS-ON domain fix.
    Real images show green/blue biased pixel speckles spatially clustered
    in dark regions. Density modulated by a low-res random field (Perlin proxy)
    via bilinear upscale. Restricted to void regions (sat_presence < 0.2).

    FIX vs V26: boolean mask (&) instead of float multiply. torch.where
    instead of direct boolean index. Added sparse R channel contribution.
    """
    c, h, w = img_tensor.shape
    density = torch.rand((1, h//4, w//4), device=img_tensor.device)
    density = F.interpolate(density.unsqueeze(0), size=(h,w),
                            mode='bilinear', align_corners=False).squeeze(0)
    sat_presence = img_tensor.max(0, keepdim=True)[0]
    void_mask    = (sat_presence < 0.2).float()
    salt_mask    = ((torch.rand((1,h,w), device=img_tensor.device) * density) > 0.98) \
                   & (void_mask > 0.5)
    n = int(salt_mask.sum().item())
    noise = torch.zeros_like(img_tensor)
    if n > 0:
        cr = torch.rand(n, device=img_tensor.device)
        r_val = torch.where(cr > 0.75,
                            torch.rand(n, device=img_tensor.device) * 0.25,
                            torch.zeros(n, device=img_tensor.device))
        noise[0, salt_mask[0]] = r_val
        noise[1, salt_mask[0]] = torch.rand(n, device=img_tensor.device) * 0.55
        noise[2, salt_mask[0]] = torch.rand(n, device=img_tensor.device) * 0.75
    return torch.clamp(img_tensor + noise, 0.0, 1.0)


def step_optical_smear(img_tensor):
    """
    RANDOM (25% prob).
    Directional gaussian smear simulating event-camera motion blur.
    Reduced kernel sizes vs V26 ([15,25,35] → [5,7,11]) to preserve
    solar panels and thin structural edges.
    """
    kx = random.choice([5, 7, 11])
    ky = random.choice([3, 5])
    if random.random() > 0.5: kx, ky = ky, kx
    glow = TF.gaussian_blur(img_tensor, [ky, kx], [ky/3.0, kx/3.0])
    return torch.clamp(img_tensor * 0.85 + glow * random.uniform(0.5, 0.9), 0.0, 1.0)


def step_vignette(img_tensor):
    """
    ALWAYS-ON domain fix.
    Corner darkening from real camera lens optics. Lower bound 0.55 ensures
    corners never go below 55% brightness (V26 had no bound, corners hit 0.3).
    """
    c, h, w = img_tensor.shape
    y = torch.linspace(-1, 1, h, device=img_tensor.device).view(h, 1)
    x = torch.linspace(-1, 1, w, device=img_tensor.device).view(1, w)
    r = torch.sqrt(x**2 + y**2)
    vig = torch.clamp(1.0 - r * random.uniform(0.10, 0.28), 0.55, 1.0)
    return torch.clamp(img_tensor * vig, 0.0, 1.0)


def step_chromatic_ab(img_tensor):
    """
    ALWAYS-ON domain fix.
    Lateral chromatic aberration: 1-2px horizontal-only R/B channel shift.
    Visible as colour fringing at bright edges in real image #3.

    FIX vs V26/V29:
      - Old: torch.roll on dims=(0,1) = DIAGONAL shift — physically wrong.
      - Old: in-place mutation of input tensor (aliasing bug).
      - New: dims=1 only (horizontal = real lateral CA). Non-mutating clone.
    """
    shift = random.randint(1, 2)
    out   = img_tensor.clone()
    out[0] = torch.roll(img_tensor[0], -shift, dims=1)  # R left
    out[2] = torch.roll(img_tensor[2],  shift, dims=1)  # B right
    return out


def step_bg_bleed(img_tensor):
    """
    RANDOM (20% prob).
    Faint diagonal line in background — facility wall seam or cable.
    Seen in real images 1, 3, 9. Blue-dominant.

    FIX vs V26/V29:
      - Old: exp(-dist/2.0) was a WIDE gradient that lit up the whole frame
        including the satellite body.
      - New: exp(-dist/0.7) is narrow (1-3px effective width).
      - New: void_mask restricts to dark pixels only (< 0.12).
      - New: blue-dominant (B×1.0, G×0.35, R×0.0).
    """
    c, h, w = img_tensor.shape
    y, x = torch.meshgrid(torch.arange(h, device=img_tensor.device),
                          torch.arange(w, device=img_tensor.device), indexing='ij')
    angle = random.uniform(-45, 45)
    cx    = random.randint(w//4, 3*w//4)
    dist  = torch.abs(  (x.float() - cx)   * math.cos(math.radians(angle))
                      - (y.float() - h//2) * math.sin(math.radians(angle)))
    streak    = torch.exp(-dist / 0.7) * random.uniform(0.05, 0.12)
    void_mask = (img_tensor.max(0)[0] < 0.12).float()
    out = img_tensor.clone()
    out[2] = torch.clamp(out[2] + streak * void_mask * 1.00, 0.0, 1.0)
    out[1] = torch.clamp(out[1] + streak * void_mask * 0.35, 0.0, 1.0)
    return out


def step_lens_flare(img_tensor):
    """
    RANDOM (15% prob).
    Radial starburst from extreme light angle — clearly visible in real image #3.
    Blue-rainbow colour matches the real chromatic flare.

    FIX vs V26/V29:
      - Old: detected satellite peaks and shot rays FROM the satellite.
        This taught the model a wrong spatial prior.
      - Old: intensity 0.7-1.2 overexposed the entire frame.
      - New: source is a random point near the frame EDGE (where the real
        studio light lives). Intensity 0.05-0.30 per ray.
    """
    c, h, w = img_tensor.shape
    side = random.choice(['left', 'right', 'top'])
    if   side == 'left':  sx, sy = random.randint(0, w//6),      random.randint(0, h//3)
    elif side == 'right': sx, sy = random.randint(5*w//6, w-1),  random.randint(0, h//3)
    else:                 sx, sy = random.randint(w//4, 3*w//4), random.randint(0, h//8)

    y, x = torch.meshgrid(torch.arange(h, device=img_tensor.device),
                          torch.arange(w, device=img_tensor.device), indexing='ij')
    flare = torch.zeros((1, h, w), device=img_tensor.device)
    for _ in range(random.randint(4, 12)):
        angle  = random.uniform(0, 2 * math.pi)
        length = random.uniform(40, 120)
        width  = random.uniform(0.4, 0.8)
        xr = (x.float()-sx)*math.cos(angle) - (y.float()-sy)*math.sin(angle)
        yr = (x.float()-sx)*math.sin(angle) + (y.float()-sy)*math.cos(angle)
        spear = torch.exp(-(torch.abs(xr)/length + torch.abs(yr)/width))
        flare[0] = torch.maximum(flare[0], spear * random.uniform(0.05, 0.30))

    # Blue-rainbow colour distribution matching real flare images
    flare_rgb = torch.cat([flare*0.25, flare*0.55, flare*1.00], dim=0)
    return torch.clamp(img_tensor + flare_rgb, 0.0, 1.0)


def step_secondary_light(img_tensor):
    """
    RANDOM (20% prob).
    Circular blue glow on right side of frame — facility secondary lamp.
    Clearly visible in real images #2 and #6.
    Intensity reduced to 0.08-0.20 (V26 was 0.2-0.4 — too bright).
    """
    c, h, w = img_tensor.shape
    cy = random.randint(h//4,   3*h//4)
    cx = random.randint(w//2,   w - 1)
    y, x = torch.meshgrid(torch.arange(h, device=img_tensor.device),
                          torch.arange(w, device=img_tensor.device), indexing='ij')
    dist = torch.sqrt((x.float()-cx)**2 + (y.float()-cy)**2)
    glow = torch.exp(-dist / random.uniform(20, 40)) * random.uniform(0.08, 0.20)
    out  = img_tensor.clone()
    out[2] = torch.clamp(out[2] + glow, 0.0, 1.0)
    out[1] = torch.clamp(out[1] + glow * 0.45, 0.0, 1.0)
    return out


def step_jpeg(img_tensor):
    """
    ALWAYS-ON, ALWAYS LAST.
    Real camera always JPEG-encodes output. Q=60-78 introduces DCT block
    artifacts at high-contrast edges, slight colour smearing, and fine
    detail softening — all present in real test data.

    FIX vs V26/V29: old version used 224→112→224 bilinear/nearest resize.
    That created pixelation artifacts (not JPEG) and destroyed fine
    satellite structure (panel edges, thin antenna features).
    """
    quality = random.randint(60, 78)
    np_img  = (img_tensor.permute(1,2,0).cpu().numpy() * 255).clip(0,255).astype(np.uint8)
    buf     = io.BytesIO()
    Image.fromarray(np_img).save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return torch.from_numpy(
        np.array(Image.open(buf)).astype(np.float32) / 255.0
    ).permute(2,0,1).to(img_tensor.device).float()


# ==============================================================================
# 2. DATASET
# ==============================================================================

class SPADESDatasetV29(Dataset):
    def __init__(self, h5_file, window_ms=400, is_train=True):
        self.h5_path  = h5_file
        self.window   = window_ms * 1000
        self.is_train = is_train
        with h5py.File(self.h5_path, 'r') as f:
            self.label_ts = f['labels/data']['timestamp'][:]
            self.tx = f['labels/data']['Tx'][:]
            self.ty = f['labels/data']['Ty'][:]
            self.tz = f['labels/data']['Tz'][:]
            self.qx = f['labels/data']['Qx'][:]
            self.qy = f['labels/data']['Qy'][:]
            self.qz = f['labels/data']['Qz'][:]
            self.qw = f['labels/data']['Qw'][:]
            self.length = len(self.label_ts)
            # off_weight computed from actual polarity distribution (not hardcoded 0.5)
            ps_samp = f['events/ps'][:500000]
            p_on    = np.mean(ps_samp)
            self.off_weight = p_on / (1.0 - p_on + 1e-6)
        self.f = None

    def __len__(self):
        return self.length

    def create_3c_tensor(self, xs, ys, ts, ps, target_ts, crop_box=None):
        img = np.zeros((3, 224, 224), dtype=np.float32)
        if len(xs) < 5:
            return torch.from_numpy(img).float(), 1.0

        weights = np.clip(1.0 - (target_ts - ts) / self.window, 0.2, 1.0)
        weights[ps == 0] *= self.off_weight

        if crop_box:
            cx, cy, sz = crop_box
            sz         = max(sz, 10.0)
            scale_hint = sz / 1280.0
            xi = np.clip((xs - (cx - sz/2)) * (224/sz), 0, 223).astype(int)
            yi = np.clip((ys - (cy - sz/2)) * (224/sz), 0, 223).astype(int)
        else:
            scale_hint = 1.0
            xi = np.clip(xs * (223/1280), 0, 223).astype(int)
            yi = np.clip(ys * (223/720),  0, 223).astype(int)

        t_step = (ts[-1] - ts[0]) / 3.0
        for i in range(3):
            m = (ts >= ts[0] + i*t_step) & (ts < ts[0] + (i+1)*t_step)
            if m.any():
                np.add.at(img[i], (yi[m], xi[m]), weights[m])

        tensor  = torch.from_numpy(img).unsqueeze(0)
        dilated = F.pad(F.max_pool2d(tensor, 2, 1, 0), (0,1,0,1))
        blur_k  = torch.ones((1,1,3,3)) / 9.0
        for ch in range(3):
            blurred = F.conv2d(dilated[:,ch:ch+1], blur_k, padding=1)
            dilated[:,ch:ch+1] = dilated[:,ch:ch+1] + 0.8*(dilated[:,ch:ch+1]-blurred)

        sharpened   = dilated.squeeze(0).numpy()
        log_img     = np.log1p(np.clip(sharpened, 0, None))
        sigmoid_img = 1.0 / (1.0 + np.exp(-10.0*(log_img/(np.max(log_img)+1e-9)-0.3)))
        final_img   = np.clip(sigmoid_img / (np.percentile(sigmoid_img, 99.5)+1e-8), 0, 1)
        return torch.from_numpy(final_img).float(), float(scale_hint)

    def apply_pipeline(self, img):
        """
        Domain-realistic augmentation pipeline.

        ALWAYS-ON (train + test):
          These are physical properties of the real camera/facility.
          Every real test image has them. Training on them at 100% is
          necessary to close the domain gap.

          1. step_blue_floor       — ambient blue floor always present
          2. step_edge_bright      — directional lighting always blows edges
          3. step_organic_noise    — sensor noise always present
          4. step_vignette         — camera lens always vignetted
          5. step_chromatic_ab     — camera optics always have some CA
          6. step_jpeg             — camera always JPEG-encodes (LAST)

        RANDOM (train only):
          These are scene-dependent phenomena that occur in some real frames
          but not all. Applied probabilistically for train diversity.

          • step_structural_debris  20%
          • step_optical_smear      25%
          • step_lens_flare         15%  ┐ mutually
          • step_secondary_light    20%  │ exclusive
          • step_bg_bleed           15%  ┘ extreme_val split
          • T.ColorJitter           always-on when train (before JPEG)
        """
        if self.is_train and random.random() < 0.20:
            return step_jpeg(img)
        out = img

        # ── ALWAYS-ON DOMAIN FIXES ────────────────────────────────────────
        out = step_blue_floor(out)
        out = step_edge_bright(out)
        out = step_organic_noise(out)
        out = step_vignette(out)
        out = step_chromatic_ab(out)

        # ── RANDOM TRAIN-ONLY AUGMENTATIONS ──────────────────────────────
        if self.is_train:
            if random.random() < 0.20:
                out =step_structural_debris(out)
            if random.random() < 0.25:
                out = step_optical_smear(out)

            # Mutually exclusive extreme events (colour jitter is separate)
            ev = random.random()
            if   ev < 0.15: out = step_lens_flare(out)
            elif ev < 0.35: out = step_secondary_light(out)
            elif ev < 0.50: out = step_bg_bleed(out)

            # ColorJitter BEFORE JPEG (colour shift happens before sensor encoding)
            out = T.ColorJitter(brightness=0.15, contrast=0.15)(out)

        # ── ALWAYS-ON POST-PROCESS (must be last) ─────────────────────────
        out = step_jpeg(out)
        return out

    def __getitem__(self, idx):
        if self.f is None:
            self.f = h5py.File(self.h5_path, 'r', swmr=True)

        target_ts = self.label_ts[idx]
        ev_ts     = self.f['events/ts']
        idx_e     = np.searchsorted(ev_ts, target_ts)
        idx_s     = np.searchsorted(ev_ts, target_ts - self.window)

        xs = self.f['events/xs'][idx_s:idx_e]
        ys = self.f['events/ys'][idx_s:idx_e]
        ts = ev_ts[idx_s:idx_e]
        ps = self.f['events/ps'][idx_s:idx_e]

        cx       = np.median(xs) if len(xs) > 0 else 640
        cy       = np.median(ys) if len(ys) > 0 else 360
        std_crop = np.clip(max(np.std(xs), np.std(ys))*12.0, 300, 1000) \
                   if len(xs) > 5 else 800

        if self.is_train and len(xs) > 0:
            cx += np.random.uniform(-std_crop*0.25, std_crop*0.25)
            cy += np.random.uniform(-std_crop*0.25, std_crop*0.25)
            if random.random() > 0.5:
                n_n = int(len(xs) * 0.40)
                xs  = np.concatenate([xs, np.random.uniform(0, 1280, n_n)])
                ys  = np.concatenate([ys, np.random.uniform(0, 720,  n_n)])
                ts  = np.concatenate([ts, np.random.uniform(ts.min(), ts.max(), n_n)])
                ps  = np.concatenate([ps, np.random.randint(0, 2, n_n)])

        img_f, _  = self.create_3c_tensor(xs, ys, ts, ps, target_ts)
        img_r, sc = self.create_3c_tensor(xs, ys, ts, ps, target_ts,
                                           crop_box=(cx, cy, std_crop))

        # Both views go through the full pipeline
        img_f = self.apply_pipeline(img_f)
        img_r = self.apply_pipeline(img_r)

        t_gt = torch.tensor([self.tx[idx], self.ty[idx], self.tz[idx]], dtype=torch.float32)
        q_gt = torch.tensor([self.qx[idx], self.qy[idx], self.qz[idx], self.qw[idx]],
                             dtype=torch.float32)
        return img_f, img_r, torch.tensor([sc]).float(), t_gt, q_gt


# ==============================================================================
# 3. MODEL — CBAM + EfficientNet-V2-S (translation) + ResNet-50 (rotation)
# ==============================================================================

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio), nn.ReLU(),
            nn.Linear(in_planes // ratio, in_planes))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a = self.fc(self.avg_pool(x).view(x.size(0), -1))
        m = self.fc(self.max_pool(x).view(x.size(0), -1))
        return self.sigmoid(a + m).view(x.size(0), x.size(1), 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1   = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))


class SparkV29Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Translation backbone: EfficientNet-V2-S
        self.backbone_t = models.efficientnet_v2_s(weights='DEFAULT').features

        # Rotation backbone: ResNet-50 + CBAM
        resnet50        = models.resnet50(weights='DEFAULT')
        self.backbone_r = nn.Sequential(*list(resnet50.children())[:-2])
        self.ca         = ChannelAttention(2048)
        self.sa         = SpatialAttention()

        # Translation head (LayerNorm for stability)
        self.t_head = nn.Sequential(
            nn.Linear(1280 + 1, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 3))

        # Rotation head (Dropout for generalisation)
        self.r_head = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 4))

    def forward(self, x_f, x_r, scale):
        # Translation: full-view EfficientNet features + scale hint
        f_t = torch.flatten(nn.AdaptiveAvgPool2d(1)(self.backbone_t(x_f)), 1)
        t   = self.t_head(torch.cat([f_t, scale.view(-1,1).to(f_t.dtype)], dim=1))

        # Rotation: cropped ResNet50 + CBAM
        f_r_map = self.backbone_r(x_r)
        f_r_map = f_r_map * self.ca(f_r_map) * self.sa(f_r_map)
        f_r     = torch.flatten(nn.AdaptiveAvgPool2d(1)(f_r_map), 1)
        q_raw   = self.r_head(f_r)
        q       = q_raw / (q_raw.norm(dim=1, keepdim=True) + 1e-8)
        return t, q


# ==============================================================================
# 4. LOSS & VALIDATION
# ==============================================================================

def spade_loss_v29(t_p, q_p, t_g, q_g):
    """
    V30 Loss: Balanced weighting for multi-task pose estimation.
    Weights rotation higher to force convergence on the harder 4-DOF problem.
    """
    # Translation: Huber Loss (beta=0.1 handles small errors with MSE, large with L1)
    t_loss = F.smooth_l1_loss(t_p, t_g, beta=0.1)
    
    # Rotation: Geodesic distance
    dot = torch.clamp(torch.abs(torch.sum(q_p * q_g, dim=1)), 0.0, 0.9999)
    r_err = torch.mean(2.0 * torch.acos(dot))
    
    # Weighting: 1.0 for Translation, 2.0 for Rotation to balance gradient magnitudes
    total_loss = t_loss + (2.0 * r_err)
    
    return total_loss, t_loss, r_err


def validate_model(model, loader, device):
    model.eval()
    val_t, val_r = [], []
    val_bias = torch.zeros((3, 224, 224), device=device)
    val_bias[1] = 0.02 # Fixed G
    val_bias[2] = 0.08 # Fixed B
    with torch.no_grad():
        for img_f, img_r, sc, t_g, q_g in loader:
            img_f  = img_f.to(device);  img_r = img_r.to(device)
            sc     = sc.to(device);     t_g   = t_g.to(device);  q_g = q_g.to(device)
            img_f = torch.max(img_f, val_bias)
            img_r = torch.max(img_r, val_bias)
            with torch.amp.autocast('cuda'):
                t_p, q_p = model(img_f, img_r, sc)
                _, lt, lr = spade_loss_v29(t_p, q_p, t_g, q_g)
            val_t.append(lt.item())
            val_r.append(np.rad2deg(lr.item()))
    return np.mean(val_t), np.mean(val_r)


# ==============================================================================
# 5. TRAINING — auto-resume, dynamic LR, NaN guard, AMP
# ==============================================================================

if __name__ == "__main__":
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_h5  = sorted(glob.glob("data/RT*.h5"))
    random.seed(42)
    random.shuffle(all_h5)

    split      = int(len(all_h5) * 0.9)
    train_ds   = ConcatDataset([SPADESDatasetV29(f, is_train=True)  for f in all_h5[:split]])
    val_ds     = ConcatDataset([SPADESDatasetV29(f, is_train=False) for f in all_h5[split:]])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)

    model     = SparkV29Net().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler    = torch.amp.GradScaler('cuda')

    # Auto-resume: finds highest epoch checkpoint in output dir
    ckpt_dir = "checkpoints_v29"
    os.makedirs(ckpt_dir, exist_ok=True)
    existing    = glob.glob(f"{ckpt_dir}/spark_v29_e*.pth")
    start_epoch = 1
    if existing:
        epochs      = [int(f.split('_e')[-1].split('.pth')[0]) for f in existing]
        latest      = max(epochs)
        latest_path = f"{ckpt_dir}/spark_v29_e{latest}.pth"
        model.load_state_dict(torch.load(latest_path, map_location=device))
        start_epoch = latest + 1
        print(f"Resumed from {latest_path} → starting epoch {start_epoch}")
    else:
        print("No checkpoint found — training from scratch.")

    for epoch in range(start_epoch, 16):
        # Dynamic LR: warm phase 1e-4 for first 3 epochs, precision phase 1e-5 after
        current_lr = 1e-4 if epoch <= 2 else 1e-5
        for pg in optimizer.param_groups:
            pg['lr'] = current_lr

        model.train()
        print(f"\nEpoch {epoch}/15 | LR: {current_lr:.0e}")

        for i, (img_f, img_r, sc, t_g, q_g) in enumerate(train_loader):
            img_f = img_f.to(device); img_r = img_r.to(device)
            sc    = sc.to(device);    t_g   = t_g.to(device);  q_g = q_g.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                t_p, q_p        = model(img_f, img_r, sc)
                loss, lt, lr    = spade_loss_v29(t_p, q_p, t_g, q_g)

            if not torch.isnan(loss):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()

            if i % 100 == 0:
                print(f"  B{i:04d} | Loss:{loss:.4f}  T:{lt.item():.4f}  "
                      f"R:{np.rad2deg(lr.item()):.1f}°")

        vt, vr = validate_model(model, val_loader, device)
        print(f"  ── Val T:{vt:.4f}  R:{vr:.2f}° ──")
        torch.save(model.state_dict(), f"{ckpt_dir}/spark_v29_e{epoch}.pth")
