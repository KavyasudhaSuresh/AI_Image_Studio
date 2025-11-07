import torch
from diffusers import StableDiffusionPipeline

# Disable autograd globally
torch.set_grad_enabled(False)

# Load model once and reuse
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float32
)
pipe.safety_checker = lambda images, clip_input: (images, [False] * len(images))
pipe.to("cpu")

def generate_image(prompt: str, output_path: str = "output.png"):
    """
    Generates an image from the given prompt using Stable Diffusion.

    Args:
        prompt (str): The text prompt to generate the image from.
        output_path (str): File path to save the output image.

    Returns:
        str: Path to the saved image.
    """
    image = pipe(prompt, num_inference_steps=20, guidance_scale=7.5).images[0]
    image.save(output_path)
    return output_path
