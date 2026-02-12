import math
import torch

class FresnelReqTorch:
    def __init__(self, d, wavelength, pixelsize, thetain_deg, dtype=torch.complex64):
        self.d = float(d)
        self.wavelength = float(wavelength)
        self.pixelsize = float(pixelsize)
        self.thetain_deg = float(thetain_deg)
        self.out_dtype = dtype
        self._cache = {}

    def _get_terms(self, H, W, device, real_dtype):
        key = (H, W, device, real_dtype)
        if key in self._cache: return self._cache[key]
        
        center = int(round((max(H,W) + 1) / 2.0))
        # Simple grid generation
        x = torch.arange(1, W + 1, device=device, dtype=real_dtype) - int(round((W + 1) / 2.0))
        y = torch.arange(1, H + 1, device=device, dtype=real_dtype) - int(round((H + 1) / 2.0))
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        pixsq = self.pixelsize ** 2
        lam = self.wavelength
        d = self.d
        theta = self.thetain_deg * math.pi / 180.0

        phase_g = math.pi * (pixsq * (xx ** 2) + pixsq * (yy ** 2)) / (lam * d)
        phase_r = (2.0 * math.pi) * xx * math.sin(theta) / lam
        phase_c = math.pi * d * lam * ((xx ** 2) / (pixsq * (W ** 2)) + (yy ** 2) / (pixsq * (H ** 2)))

        g = torch.exp(1j * phase_g).to(self.out_dtype)
        r = torch.exp(1j * phase_r).to(self.out_dtype)
        c = torch.exp(1j * phase_c).to(self.out_dtype)
        
        self._cache[key] = (g, r, c)
        return g, r, c

    def __call__(self, I):
        if I.ndim == 2: I = I[None, :, :]
        N, H, W = I.shape
        device = I.device
        real_dtype = torch.float32 if I.dtype in (torch.float32, torch.complex64) else torch.float64
        
        g, r, c = self._get_terms(H, W, device, real_dtype)
        
        h = I.to(self.out_dtype)
        b = h * g * r
        
        # Pure GPU FFT
        Rh = torch.fft.fftshift(torch.fft.fft2(b), dim=(-2, -1))
        
        return Rh * c