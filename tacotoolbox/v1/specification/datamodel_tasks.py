from typing_extensions import Literal, TypeAlias

Task: TypeAlias = Literal[
    "regression",
    "classification",
    "scene-classification",
    "detection",
    "object-detection",
    "segmentation",
    "semantic-segmentation",
    "instance-segmentation",
    "panoptic-segmentation",
    "similarity-search",
    "generative",
    "image-captioning",
    "super-resolution",
    "denoising",
    "inpainting",
    "colorization",
    "style-transfer",
    "deblurring",
    "dehazing",
    "general",
]
