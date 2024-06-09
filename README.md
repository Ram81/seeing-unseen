# Seeing the Unseen: Visual Common Sense for Semantic Placement


Code for our paper **[Seeing the Unseen: Visual Common Sense for Semantic Placement](https://arxiv.org/pdf/2401.07770.pdf)**

Ram Ramrakhya, Aniruddha Kembhavi, Dhruv Batra, Zsolt Kira, Kuo-Hao Zeng*, Luca Weihs*


<p align="center">
  <img src="imgs/idea.gif"  height="400">

  <p align="center">Approach: Leverage advances in vision foundation model and diffusion models to automatically generate paired training data at scale using images in the wild.</p>
</p>

<p align="center">
    <a href="https://ram81.github.io/projects/seeing-unseen.html">Project Page</a>
</p>

Computer vision tasks typically involve describing what is visible in an image (e.g. classification, detection, segmentation, and captioning). We study a visual common sense task that requires understanding 'what is not visible'. Specifically, given an image (e.g. of a living room) and a name of an object ("cushion"), a vision system is asked to predict semantically-meaningful regions (masks or bounding boxes) in the image where that object could be placed or is likely be placed by humans (e.g. on the sofa). We call this task: Semantic Placement (SP) and believe that such common-sense visual understanding is critical for assitive robots (tidying a house), AR devices (automatically rendering an object in the user's space), and visually-grounded chatbots with common sense. Studying the invisible is hard. Datasets for image description are typically constructed by curating relevant images (e.g. via image search with object names) and asking humans to annotate the contents of the image; neither of those two steps are straightforward for objects not present in the image. We overcome this challenge by operating in the opposite direction: we start with an image of an object in context (which is easy to find online) and remove that object from the image via inpainting. This automated pipeline converts unstructured web data into a paired with/without object dataset. With this proposed data generation pipeline, we collect a novel dataset, containing ~1.3M images across 9 object categories. We then train a SP prediction model, called CLIP-UNet, on our dataset. The CLIP-UNet outperforms existing VLMs and baselines that combine semantic priors with object detectors, generalizes well to real-world and simulated images, exhibits semantics-aware reasoning for object placement, and enables downstream applications like tidying robots in indoor environments.


## :hammer: Installation

Run the provided `setup.sh` which contains relevant commands to install all required packages for running the codebase.

```bash
bash setup.sh
```


## :floppy_disk: Dataset

Download the Semantic Placement dataset generated using LAION-400M images from the following [huggingface repo 🤗]() using following command:

```bash
git clone https://huggingface.co/datasets/axel81/seeing-unseen data/datasets/
```

This command will download the full dataset with $~1.3$ million images for $9$ object categories.


## :bar_chart: Training


### CLIP-UNet baseline

Run the following command to train the CLIP-UNet model that uses frozen CLIP text and image encoder and learns a UNet decoder for predicting Semantic Placement:

```bash
python seeing_unseen/run.py config/baseline/clip_unet.yaml \
  run_type=train \
  training.epochs=25 \
  training.batch_size=32 \
  training.lr=0.0003 \
  dataset.root_dir="data/datasets/semantic_placement" \
  checkpoint_dir="/path/to/checkpoint/dir/"
```



## 🎯 Evaluation

### CLIP-UNet

Use the following command to evaluate CLIP-UNet on Semantic Placement:

```bash
python seeing_unseen/run.py   \
  run_type=eval  \
  dataset.root_dir="/path/to/val/split/"   \
  checkpoint_dir="/path/to/checkpoint/" \
  model.name="clip_unet" \
  training.eval_splits="['val']" \
  training.eval_with_tta=false \
  training.trainer="semantic_placement_evaluator"
```

### LLaVA

Use the following command to evaluate LLaVA on Semantic Placement:

```bash
python seeing_unseen/run.py   \
  run_type=eval  \
  dataset.root_dir="/path/to/val/split/"   \
  checkpoint_dir="/path/to/checkpoint/" \
  model.name="llava" \
  training.eval_splits="['val']" \
  training.eval_with_tta=false \
  training.trainer="semantic_placement_evaluator"
```

### LLM + Detector

Use the following command to evaluate LLM+Detector baseline using Detic detector on Semantic Placement:

```bash
python seeing_unseen/run.py   \
  run_type=eval  \
  dataset.root_dir="/path/to/val/split/"   \
  checkpoint_dir="/path/to/checkpoint/" \
  model.name="llm_detect_detic" \
  training.eval_splits="['val']" \
  training.eval_with_tta=false \
  training.trainer="semantic_placement_evaluator"
```


## :pencil: Citation

If you use this code, dataset, or task in your research, please consider citing:

```
@inproceedings{ramrakhya2024seeing,
  title={Seeing the Unseen: Visual Common Sense for Semantic Placement},
  author={Ram Ramrakhya and Aniruddha Kembhavi and Dhruv Batra and Zsolt Kira and Kuo-Hao Zeng and Luca Weihs},
  year={2024},
  booktitle={CVPR},
}
```