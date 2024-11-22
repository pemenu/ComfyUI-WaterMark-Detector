import torch
import numpy as np
import cv2
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import pywt

class CLAHEEnhancement:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), 
                             "clip_limit": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                             "grid_size": ("INT", {"default": 8, "min": 2, "max": 16, "step": 1})}}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "enhance"
    CATEGORY = "image/watermark"

    def enhance(self, image, clip_limit, grid_size):
        result = []
        for img in image:
            img_np = np.clip(255. * img.cpu().numpy(), 0, 255).astype(np.uint8)
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
            result.append(torch.from_numpy(enhanced.astype(np.float32) / 255.0))
        return (torch.stack(result),)

class HighPassFilter:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), 
                             "cutoff_freq": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1})}}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "filter"
    CATEGORY = "image/watermark"

    def filter(self, image, cutoff_freq):
        result = []
        for img in image:
            img_np = np.clip(255. * img.cpu().numpy(), 0, 255).astype(np.uint8)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            f = fft2(gray)
            fshift = fftshift(f)
            rows, cols = gray.shape
            crow, ccol = rows // 2, cols // 2
            mask = np.ones((rows, cols), np.uint8)
            mask[crow-cutoff_freq:crow+cutoff_freq, ccol-cutoff_freq:ccol+cutoff_freq] = 0
            fshift = fshift * mask
            f_ishift = ifftshift(fshift)
            img_back = np.abs(ifft2(f_ishift))
            img_back = (img_back - np.min(img_back)) / (np.max(img_back) - np.min(img_back))
            enhanced = np.repeat(img_back[:,:,np.newaxis], 3, axis=2)
            result.append(torch.from_numpy(enhanced.astype(np.float32)))
        return (torch.stack(result),)

class EdgeDetection:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), 
                             "low_threshold": ("INT", {"default": 50, "min": 0, "max": 255, "step": 1}),
                             "high_threshold": ("INT", {"default": 150, "min": 0, "max": 255, "step": 1})}}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "detect"
    CATEGORY = "image/watermark"

    def detect(self, image, low_threshold, high_threshold):
        result = []
        for img in image:
            img_np = np.clip(255. * img.cpu().numpy(), 0, 255).astype(np.uint8)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, low_threshold, high_threshold)
            edges_rgb = np.repeat(edges[:,:,np.newaxis], 3, axis=2)
            result.append(torch.from_numpy(edges_rgb.astype(np.float32) / 255.0))
        return (torch.stack(result),)

class CombineEnhancements:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"original": ("IMAGE",), 
                             "enhanced1": ("IMAGE",), 
                             "enhanced2": ("IMAGE",), 
                             "enhanced3": ("IMAGE",),
                             "weight1": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05}),
                             "weight2": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05}),
                             "weight3": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05})}}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "combine"
    CATEGORY = "image/watermark"

    def combine(self, original, enhanced1, enhanced2, enhanced3, weight1, weight2, weight3):
        result = []
        for orig, enh1, enh2, enh3 in zip(original, enhanced1, enhanced2, enhanced3):
            combined = orig * (1 - weight1 - weight2 - weight3) + enh1 * weight1 + enh2 * weight2 + enh3 * weight3
            result.append(torch.clamp(combined, 0, 1))
        return (torch.stack(result),)

class AdaptiveThresholding:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), 
                             "block_size": ("INT", {"default": 15, "min": 3, "max": 99, "step": 2}),
                             "C": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.5})}}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "threshold"
    CATEGORY = "image/watermark"

    def threshold(self, image, block_size, C):
        result = []
        for img in image:
            img_np = np.clip(255. * img.cpu().numpy(), 0, 255).astype(np.uint8)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)
            thresh_rgb = np.repeat(thresh[:,:,np.newaxis], 3, axis=2)
            result.append(torch.from_numpy(thresh_rgb.astype(np.float32) / 255.0))
        return (torch.stack(result),)

class MorphologicalOperations:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), 
                             "operation": (["dilate", "erode", "open", "close"],),
                             "kernel_size": ("INT", {"default": 5, "min": 1, "max": 21, "step": 2})}}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "morph"
    CATEGORY = "image/watermark"

    def morph(self, image, operation, kernel_size):
        result = []
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        for img in image:
            img_np = np.clip(255. * img.cpu().numpy(), 0, 255).astype(np.uint8)
            if operation == "dilate":
                morphed = cv2.dilate(img_np, kernel, iterations=1)
            elif operation == "erode":
                morphed = cv2.erode(img_np, kernel, iterations=1)
            elif operation == "open":
                morphed = cv2.morphologyEx(img_np, cv2.MORPH_OPEN, kernel)
            elif operation == "close":
                morphed = cv2.morphologyEx(img_np, cv2.MORPH_CLOSE, kernel)
            result.append(torch.from_numpy(morphed.astype(np.float32) / 255.0))
        return (torch.stack(result),)

class ImprovedGrayColorEnhancement:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "lower_gray": ("INT", {"default": 80, "min": 0, "max": 255, "step": 1}),
                             "upper_gray": ("INT", {"default": 220, "min": 0, "max": 255, "step": 1}),
                             "boost_factor": ("FLOAT", {"default": 1.8, "min": 1.0, "max": 5.0, "step": 0.1}),
                             "sharpen_amount": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 5.0, "step": 0.1})}}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "enhance_gray"
    CATEGORY = "image/watermark"

    def enhance_gray(self, image, lower_gray, upper_gray, boost_factor, sharpen_amount):
        result = []
        for img in image:
            img_np = np.clip(255. * img.cpu().numpy(), 0, 255).astype(np.uint8)
            
            # Convert to LAB color space
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Create a mask for gray areas using A and B channels
            gray_mask = cv2.inRange(lab, (0, 128-10, 128-10), (255, 128+10, 128+10))
            
            # Boost the L channel of gray areas
            l[gray_mask > 0] = np.clip(l[gray_mask > 0] * boost_factor, 0, 255)
            
            # Merge the channels back
            lab_boosted = cv2.merge([l, a, b])
            
            # Convert back to RGB
            enhanced = cv2.cvtColor(lab_boosted, cv2.COLOR_LAB2RGB)
            
            # Apply sharpening
            blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
            enhanced = cv2.addWeighted(enhanced, 1 + sharpen_amount, blurred, -sharpen_amount, 0)
            
            result.append(torch.from_numpy(enhanced.astype(np.float32) / 255.0))
        return (torch.stack(result),)

class TextureEnhancement:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "frequency_range": ("INT", {"default": 40, "min": 1, "max": 100, "step": 1}),
                             "boost_factor": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 5.0, "step": 0.1})}}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "enhance_texture"
    CATEGORY = "image/watermark"

    def enhance_texture(self, image, frequency_range, boost_factor):
        result = []
        for img in image:
            img_np = np.clip(255. * img.cpu().numpy(), 0, 255).astype(np.uint8)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            # Perform FFT
            f = fft2(gray)
            fshift = fftshift(f)
            
            # Create a band-pass filter
            rows, cols = gray.shape
            crow, ccol = rows // 2, cols // 2
            mask = np.zeros((rows, cols), np.float32)
            mask[crow-frequency_range:crow+frequency_range, ccol-frequency_range:ccol+frequency_range] = 1
            mask[crow-frequency_range//2:crow+frequency_range//2, ccol-frequency_range//2:ccol+frequency_range//2] = 0
            
            # Apply filter and inverse FFT
            fshift_filtered = fshift * mask
            f_ishift = ifftshift(fshift_filtered)
            img_back = np.abs(ifft2(f_ishift))
            
            # Normalize and boost
            img_back = (img_back - np.min(img_back)) / (np.max(img_back) - np.min(img_back))
            img_back = np.clip(img_back * boost_factor, 0, 1)
            
            # Combine with original image
            enhanced = img_np.astype(np.float32) / 255.0
            enhanced += np.repeat(img_back[:,:,np.newaxis], 3, axis=2)
            enhanced = np.clip(enhanced, 0, 1)
            
            result.append(torch.from_numpy(enhanced.astype(np.float32)))
        return (torch.stack(result),)

class DenoisingFilter:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "strength": ("FLOAT", {"default": 8, "min": 0, "max": 20, "step": 0.1}),
                             "color_strength": ("FLOAT", {"default": 8, "min": 0, "max": 20, "step": 0.1})}}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "denoise"
    CATEGORY = "image/watermark"

    def denoise(self, image, strength, color_strength):
        result = []
        for img in image:
            img_np = np.clip(255. * img.cpu().numpy(), 0, 255).astype(np.uint8)
            denoised = cv2.fastNlMeansDenoisingColored(img_np, None, strength, color_strength, 7, 21)
            result.append(torch.from_numpy(denoised.astype(np.float32) / 255.0))
        return (torch.stack(result),)

class FlexibleCombineEnhancements:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"original": ("IMAGE",)},
                "optional": {f"enhanced{i}": ("IMAGE",) for i in range(1, 6)},
                "hidden": {f"weight{i}": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05}) for i in range(1, 6)}}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "combine"
    CATEGORY = "image/watermark"

    def combine(self, original, **kwargs):
        enhanced_images = [kwargs.get(f"enhanced{i}") for i in range(1, 6) if f"enhanced{i}" in kwargs]
        weights = [kwargs.get(f"weight{i}", 0.2) for i in range(1, 6) if f"enhanced{i}" in kwargs]
        
        result = []
        for imgs in zip(original, *enhanced_images):
            combined = imgs[0] * (1 - sum(weights))
            for img, weight in zip(imgs[1:], weights):
                combined += img * weight
            result.append(torch.clamp(combined, 0, 1))
        return (torch.stack(result),)

class ComprehensiveImageEnhancement:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                # CLAHE Enhancement
                "clahe_clip_limit": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "clahe_grid_size": ("INT", {"default": 8, "min": 2, "max": 16, "step": 1}),
                # High Pass Filter
                "hpf_cutoff_freq": ("INT", {"default": 80, "min": 1, "max": 100, "step": 1}),
                # Edge Detection
                "edge_low_threshold": ("INT", {"default": 30, "min": 0, "max": 255, "step": 1}),
                "edge_high_threshold": ("INT", {"default": 100, "min": 0, "max": 255, "step": 1}),
                # Adaptive Thresholding
                "at_block_size": ("INT", {"default": 11, "min": 3, "max": 99, "step": 2}),
                "at_c": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.5}),
                # Morphological Operations
                "morph_operation": (["dilate", "erode", "open", "close"],),
                "morph_kernel_size": ("INT", {"default": 3, "min": 1, "max": 21, "step": 2}),
                # Improved Gray Color Enhancement
                "gray_lower": ("INT", {"default": 50, "min": 0, "max": 255, "step": 1}),
                "gray_upper": ("INT", {"default": 200, "min": 0, "max": 255, "step": 1}),
                "gray_boost_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 5.0, "step": 0.1}),
                "gray_sharpen_amount": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 5.0, "step": 0.1}),
                # Texture Enhancement
                "texture_freq_range": ("INT", {"default": 70, "min": 1, "max": 100, "step": 1}),
                "texture_boost_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 5.0, "step": 0.1}),
                # Denoising Filter
                "denoise_strength": ("FLOAT", {"default": 5, "min": 0, "max": 20, "step": 0.1}),
                "denoise_color_strength": ("FLOAT", {"default": 5, "min": 0, "max": 20, "step": 0.1}),
                # Weights for combining
                "weight_clahe": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "weight_hpf": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05}),
                "weight_edge": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05}),
                "weight_at": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
                "weight_morph": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
                "weight_gray": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "weight_texture": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "weight_denoise": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "enhance"
    CATEGORY = "image/watermark"

    def enhance(self, image, **kwargs):
        clahe = CLAHEEnhancement()
        hpf = HighPassFilter()
        edge = EdgeDetection()
        at = AdaptiveThresholding()
        morph = MorphologicalOperations()
        gray = ImprovedGrayColorEnhancement()
        texture = TextureEnhancement()
        denoise = DenoisingFilter()
        
        enhanced_clahe = clahe.enhance(image, kwargs['clahe_clip_limit'], kwargs['clahe_grid_size'])[0]
        enhanced_hpf = hpf.filter(image, kwargs['hpf_cutoff_freq'])[0]
        enhanced_edge = edge.detect(image, kwargs['edge_low_threshold'], kwargs['edge_high_threshold'])[0]
        enhanced_at = at.threshold(image, kwargs['at_block_size'], kwargs['at_c'])[0]
        enhanced_morph = morph.morph(image, kwargs['morph_operation'], kwargs['morph_kernel_size'])[0]
        enhanced_gray = gray.enhance_gray(image, kwargs['gray_lower'], kwargs['gray_upper'], 
                                          kwargs['gray_boost_factor'], kwargs['gray_sharpen_amount'])[0]
        enhanced_texture = texture.enhance_texture(image, kwargs['texture_freq_range'], kwargs['texture_boost_factor'])[0]
        enhanced_denoise = denoise.denoise(image, kwargs['denoise_strength'], kwargs['denoise_color_strength'])[0]
        
        result = []
        for imgs in zip(image, enhanced_clahe, enhanced_hpf, enhanced_edge, enhanced_at, 
                        enhanced_morph, enhanced_gray, enhanced_texture, enhanced_denoise):
            original = imgs[0]
            enhancements = imgs[1:]
            weights = [kwargs[f'weight_{name}'] for name in ['clahe', 'hpf', 'edge', 'at', 'morph', 'gray', 'texture', 'denoise']]
            
            combined = original * (1 - sum(weights))
            for enhanced, weight in zip(enhancements, weights):
                combined += enhanced * weight
            
            result.append(torch.clamp(combined, 0, 1))
        
        return (torch.stack(result),)

class WatermarkEnhancement:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "contrast_factor": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1}),
                             "sharpness_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 5.0, "step": 0.1})}}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "enhance_watermark"
    CATEGORY = "image/watermark"

    def enhance_watermark(self, image, contrast_factor, sharpness_factor):
        result = []
        for img in image:
            img_np = np.clip(255. * img.cpu().numpy(), 0, 255).astype(np.uint8)
            
            # Increase contrast
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
            
            # Apply sharpening
            blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
            enhanced = cv2.addWeighted(enhanced, sharpness_factor, blurred, 1-sharpness_factor, 0)
            
            # Adjust contrast
            enhanced = cv2.convertScaleAbs(enhanced, alpha=contrast_factor, beta=0)
            
            result.append(torch.from_numpy(enhanced.astype(np.float32) / 255.0))
        return (torch.stack(result),)

class AdvancedWatermarkEnhancement:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "method": (["fourier", "wavelet", "phase_congruency", "adaptive_threshold"],),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
        }}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "enhance_watermark"
    CATEGORY = "image/watermark"

    def enhance_watermark(self, image, method, strength):
        result = []
        for img in image:
            img_np = np.clip(255. * img.cpu().numpy(), 0, 255).astype(np.uint8)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

            if method == "fourier":
                enhanced = self.fourier_enhance(gray, strength)
            elif method == "wavelet":
                enhanced = self.wavelet_enhance(gray, strength)
            elif method == "phase_congruency":
                enhanced = self.phase_congruency_enhance(gray, strength)
            elif method == "adaptive_threshold":
                enhanced = self.adaptive_threshold_enhance(gray, strength)

            enhanced_rgb = np.stack([enhanced] * 3, axis=-1)
            result.append(torch.from_numpy(enhanced_rgb.astype(np.float32) / 255.0))
        return (torch.stack(result),)

    def fourier_enhance(self, img, strength):
        f = fft2(img)
        fshift = fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift))
        
        # Enhance high frequencies
        rows, cols = img.shape
        crow, ccol = rows // 2, cols // 2
        fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
        
        f_ishift = ifftshift(fshift)
        img_back = np.abs(ifft2(f_ishift))
        img_back = (img_back - np.min(img_back)) / (np.max(img_back) - np.min(img_back))
        return (img_back * 255).astype(np.uint8)

    def wavelet_enhance(self, img, strength):
        coeffs2 = pywt.dwt2(img, 'bior1.3')
        LL, (LH, HL, HH) = coeffs2
        
        # Enhance high frequency components
        LH = LH * strength
        HL = HL * strength
        HH = HH * strength
        
        enhanced_coeffs = LL, (LH, HL, HH)
        enhanced_img = pywt.idwt2(enhanced_coeffs, 'bior1.3')
        return np.clip(enhanced_img, 0, 255).astype(np.uint8)

    def phase_congruency_enhance(self, img, strength):
        # This is a simplified version. For full implementation, consider using the phasepack library
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        phase = np.arctan2(sobely, sobelx)
        
        enhanced = magnitude * np.cos(phase) * strength
        return np.clip(enhanced, 0, 255).astype(np.uint8)

    def adaptive_threshold_enhance(self, img, strength):
        block_size = int(11 + strength * 10)  # Adjust block size based on strength
        C = strength * 2
        binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)
        enhanced = cv2.addWeighted(img, 1, binary, strength, 0)
        return enhanced.astype(np.uint8)

class AdvancedWaveletWatermarkEnhancement:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "wavelet_level": ("INT", {"default": 2, "min": 1, "max": 4, "step": 1}),
            "detail_enhancement": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 5.0, "step": 0.1}),
            "contrast_enhancement": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1}),
            "sharpening_amount": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 3.0, "step": 0.1}),
        }}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "enhance_watermark"
    CATEGORY = "image/watermark"

    def enhance_watermark(self, image, wavelet_level, detail_enhancement, contrast_enhancement, sharpening_amount):
        result = []
        for img in image:
            img_np = np.clip(255. * img.cpu().numpy(), 0, 255).astype(np.uint8)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

            # Apply wavelet transform
            enhanced = self.wavelet_enhance(gray, wavelet_level, detail_enhancement)

            # Apply contrast enhancement
            enhanced = self.contrast_enhance(enhanced, contrast_enhancement)

            # Apply sharpening
            enhanced = self.sharpen(enhanced, sharpening_amount)

            # Apply adaptive thresholding
            enhanced = self.adaptive_threshold(enhanced)

            enhanced_rgb = np.stack([enhanced] * 3, axis=-1)
            result.append(torch.from_numpy(enhanced_rgb.astype(np.float32) / 255.0))
        return (torch.stack(result),)

    def wavelet_enhance(self, img, level, enhancement_factor):
        coeffs = pywt.wavedec2(img, 'haar', level=level)
        
        # Enhance detail coefficients
        for i in range(1, len(coeffs)):
            coeffs[i] = tuple(enhancement_factor * detail for detail in coeffs[i])
        
        # Reconstruct the image
        enhanced_img = pywt.waverec2(coeffs, 'haar')
        return np.clip(enhanced_img, 0, 255).astype(np.uint8)

    def contrast_enhance(self, img, factor):
        mean = np.mean(img)
        enhanced = (img - mean) * factor + mean
        return np.clip(enhanced, 0, 255).astype(np.uint8)

    def sharpen(self, img, amount):
        blurred = cv2.GaussianBlur(img, (0, 0), 3)
        sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def adaptive_threshold(self, img):
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "threshold_method": (["otsu", "adaptive", "manual"],),
            "manual_threshold": ("INT", {"default": 127, "min": 0, "max": 255, "step": 1}),
            "block_size": ("INT", {"default": 11, "min": 3, "max": 99, "step": 2}),
            "constant": ("INT", {"default": 2, "min": -10, "max": 10, "step": 1}),
            "denoise_iterations": ("INT", {"default": 2, "min": 0, "max": 10, "step": 1}),
            "edge_enhance": ("BOOLEAN", {"default": True}),
        }}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "detect_watermark"
    CATEGORY = "image/watermark"

    def detect_watermark(self, image, threshold_method, manual_threshold, block_size, constant, denoise_iterations, edge_enhance):
        result = []
        for img in image:
            img_np = np.clip(255. * img.cpu().numpy(), 0, 255).astype(np.uint8)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            # Apply thresholding
            if threshold_method == "otsu":
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif threshold_method == "adaptive":
                binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY, block_size, constant)
            else:  # manual
                _, binary = cv2.threshold(gray, manual_threshold, 255, cv2.THRESH_BINARY)
            
            # Denoise
            if denoise_iterations > 0:
                binary = self.denoise_binary(binary, denoise_iterations)
            
            # Edge enhancement
            if edge_enhance:
                binary = self.enhance_edges(binary)
            
            # Convert back to RGB for consistency
            binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
            result.append(torch.from_numpy(binary_rgb.astype(np.float32) / 255.0))
        
        return (torch.stack(result),)

    def denoise_binary(self, binary, iterations):
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        return binary

    def enhance_edges(self, binary):
        edges = cv2.Canny(binary, 100, 200)
        return cv2.addWeighted(binary, 1, edges, 0.5, 0)

NODE_CLASS_MAPPINGS = {
    "CLAHEEnhancement": CLAHEEnhancement,
    "HighPassFilter": HighPassFilter,
    "EdgeDetection": EdgeDetection,
    "CombineEnhancements": CombineEnhancements,
    "AdaptiveThresholding": AdaptiveThresholding,
    "MorphologicalOperations": MorphologicalOperations,
    "ImprovedGrayColorEnhancement": ImprovedGrayColorEnhancement,
    "TextureEnhancement": TextureEnhancement,
    "DenoisingFilter": DenoisingFilter,
    "FlexibleCombineEnhancements": FlexibleCombineEnhancements,
    "ComprehensiveImageEnhancement": ComprehensiveImageEnhancement,
    "WatermarkEnhancement": WatermarkEnhancement,
    "AdvancedWatermarkEnhancement": AdvancedWatermarkEnhancement,
    "AdvancedWaveletWatermarkEnhancement": AdvancedWaveletWatermarkEnhancement,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLAHEEnhancement": "CLAHE Enhancement",
    "HighPassFilter": "High Pass Filter",
    "EdgeDetection": "Edge Detection",
    "CombineEnhancements": "Combine Enhancements",
    "AdaptiveThresholding": "Adaptive Thresholding",
    "MorphologicalOperations": "Morphological Operations",
    "ImprovedGrayColorEnhancement": "Improved Gray Color Enhancement",
    "TextureEnhancement": "Texture Enhancement",
    "DenoisingFilter": "Denoising Filter",
    "FlexibleCombineEnhancements": "Flexible Combine Enhancements",
    "ComprehensiveImageEnhancement": "Comprehensive Image Enhancement",
    "WatermarkEnhancement": "Watermark Enhancement",
    "AdvancedWatermarkEnhancement": "Advanced Watermark Enhancement",
    "AdvancedWaveletWatermarkEnhancement": "Advanced Wavelet Watermark Enhancement",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']