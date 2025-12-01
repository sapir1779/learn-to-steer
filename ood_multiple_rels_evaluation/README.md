# Out-of-Distribution and Multiple Spatial Relations Evaluation for Vision-Language Models

Evaluation framework for assessing vision-language model performance on compositional visual relationships, with support for spatial and semantic constraints, which includes multiple spatial relations in the same prompt.

## Overview

This project evaluates how well text-to-image models understand and generate scenes with specific compositional requirements (object presence, single/multiple spatial relationships). It uses CLIPSeg for object detection and relative position estimation based on bounding boxes.

## Installation

```bash
conda create -n ood_eval python=3.8 -y
conda activate ood_eval
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Usage
Download example of an image directory from [here](https://drive.google.com/drive/folders/1ZNd_qZLOP20zKD_WU1RTwiGaF347dr0C?usp=sharing).

```bash
python evaluate_images.py <image_directory> [--gpu_indices 0]
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{yiflach2026data,
  title={Data-Driven Loss Functions for Inference-Time Optimization in Text-to-Image},
  author={Yiflach, Sapir Esther and Atzmon, Yuval and Chechik, Gal},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={3525--3535},
  year={2026}
}
```

## Acknowledgments

- Built on the [GenEval](https://github.com/djghosh13/geneval) framework
- Uses [CLIPSeg](https://github.com/timojl/clipseg) for object detection

## License

See [LICENSE](LICENSE) file for details.
