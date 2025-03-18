import os
import torch
import numpy as np
from PIL import Image, ImageOps
from torchvision.transforms.v2 import ToPILImage
import chainer
from chainer import cuda, serializers, Variable

try:
    from .net import Generator
except ImportError:
    print("Error importing Generator class")

def pil2tensor(images: Image.Image | list[Image.Image]) -> torch.Tensor:
    """Converts a PIL Image or a list of PIL Images to a tensor."""
  
    def single_pil2tensor(image: Image.Image) -> torch.Tensor:
        np_image = np.array(image).astype(np.float32) / 255.0
        if np_image.ndim == 2:  # Grayscale
            np_image = np.expand_dims(np_image, 0)  # (1, H, W)
            return torch.from_numpy(np_image)
        else:  # RGB or RGBA
            np_image = np.transpose(np_image, (2, 0, 1))  # (C, H, W)
            return torch.from_numpy(np_image)
  
    if isinstance(images, Image.Image):
        return single_pil2tensor(images).unsqueeze(0)  # Add batch dimension
    else:
        return torch.cat([single_pil2tensor(img).unsqueeze(0) for img in images], dim=0)

class SketchSimplifier:
    """将粗略草图简化为干净线条的节点"""
    
    def __init__(self):
        self.model = None
        self.initialized = False
        self.to_pil = ToPILImage()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "use_gpu": (["yes", "no"], {"default": "no"}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "simplify_sketch"
    CATEGORY = "image/processing"
    
    def initialize_model(self, use_gpu="no"):
        """初始化模型，只需要在第一次运行时执行"""
        if self.initialized:
            return
            
        self.model = Generator()
        
        # 获取模型文件路径
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                  "models", "model_iter_39000.npz")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        serializers.load_npz(model_path, self.model)
        
        # 如果使用GPU，将模型移至GPU
        if use_gpu == "yes" and cuda.available:
            cuda.get_device(0).use()  # 使用第一个GPU
            self.model.to_gpu()
            
        self.initialized = True
        
    def tensor2pil(self, img_tensor):
        """将tensor转换为PIL图像，使用PyTorch的ToPILImage"""
        with torch.no_grad():
            # ComfyUI中的图像格式通常是[B, H, W, C]，需要转换为[B, C, H, W]
            # 然后取第一帧[0]，得到[C, H, W]
            pil_image = self.to_pil(img_tensor.permute(0, 3, 1, 2)[0]).convert("RGB")
            return pil_image
    
    def load_image(self, img_tensor, scale=1.0):
        """从tensor加载图像并预处理"""
        # 将tensor转为PIL图像
        pil_img = self.tensor2pil(img_tensor.unsqueeze(0))  # 添加batch维度
        
        # 转为灰度图
        if pil_img.mode != 'L':
            pil_img = pil_img.convert('L')
        
        pil_img = ImageOps.autocontrast(pil_img, 0)
        
        # 调整大小
        try:
            new_width = int(scale * pil_img.width)
            new_height = int(scale * pil_img.height)
            pil_img = pil_img.resize((new_width, new_height))
            print(f"Image resized to: {new_width}x{new_height}")
        except Exception as e:
            print(f"Error resizing image: {e}")
        
        # 转换为numpy数组并预处理
        img = np.asarray(pil_img, dtype=np.float32)
        original_shape = img.shape
        print(f"Original shape before padding: {original_shape}")
        
        img2 = np.pad(img, ((31, 31), (31, 31)), 'edge')
        img2 = img2[np.newaxis, np.newaxis, :, :] / 127.5 - 1
        
        return img2, original_shape
        
    def simplify(self, img, use_gpu="no"):
        """使用模型简化草图"""
        if use_gpu == "yes" and cuda.available:
            img = cuda.to_gpu(img)
            
        img = Variable(img)
        
        with chainer.no_backprop_mode():
            with chainer.using_config('train', False):
                gen = self.model(img)
                
        gen = gen.data
        
        if use_gpu == "yes" and cuda.available:
            gen = cuda.to_cpu(gen)
            
        return gen
    
    def simplify_sketch(self, image, use_gpu="no", scale=1.0):
        """主函数，处理输入图像并输出简化后的图像"""
        # 初始化模型
        self.initialize_model(use_gpu)
        
        # 处理每一帧图像
        result_images = []
        
        for i in range(image.shape[0]):
            # 加载和预处理图像
            img_tensor = image[i]
            img, shape = self.load_image(img_tensor, scale)
            
            # 简化图像
            gen = self.simplify(img, use_gpu)
            
            # 将生成的图像数据处理成普通图像
            processed_img = (gen[0][0] + 1) * 127.5
            processed_img = np.uint8(processed_img[31:shape[0]+31, 31:shape[1]+31])
            
            # 转为PIL图像
            pil_img = Image.fromarray(processed_img, mode='L')
            print(f"Generated image size: {pil_img.width}x{pil_img.height}")
            
            # 确保结果是RGB模式
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            
            result_images.append(pil_img)
        
        # 将PIL图像列表转换为tensor
        result_tensor = pil2tensor(result_images)
        
        # 转换格式以符合ComfyUI的期望
        # ComfyUI中的图像格式通常是[B, H, W, C]，而我们的tensor此时是[B, C, H, W]
        result_tensor = result_tensor.permute(0, 2, 3, 1)
        
        print(f"Final tensor shape: {result_tensor.shape}")
        return (result_tensor,)

# 这个函数用于在ComfyUI中注册我们的节点
NODE_CLASS_MAPPINGS = {
    "SketchSimplifier": SketchSimplifier
}

# 这个变量用于定义节点在UI中显示的名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "SketchSimplifier": "Sketch Simplifier"
}