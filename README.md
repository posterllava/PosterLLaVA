# PosterLLaVa: Constructing a Unified Multi-modal Layout Generator with LLM
This repository is for the paper "PosterLLaVa: Constructing a Unified Multi-modal Layout Generator with LLM" (under review).

<div align="center"><img src="/framework.png" width="800"></div>

## 🎊 News: Automatic Text-to-poster pipeline PosterGen
Now we are heading from layout toward the real sense of *graphic poster* design. A *text-to-poster* pipeline, **PosterGen**, will soon be available online to support the real-world application of PosterLLaVA. There are some previewing examples:

<div align="center">
      <img src="https://github.com/user-attachments/assets/a8c7e293-234b-4114-b188-dbf9590a726f" width="150">
      <img src="https://github.com/user-attachments/assets/723239b4-7f98-40e7-a375-0daef841b6a2" width="150">
      <img src="https://github.com/user-attachments/assets/bc9c5be5-6fa5-422f-9c7d-f94d1a4be54d" width="150">
      <img src="https://github.com/user-attachments/assets/b78df9b9-4553-4ac8-b823-078844e11ef9" width="150">
      <img src="https://github.com/user-attachments/assets/d63daf39-167c-4599-84eb-b8551f234a1e" width="150">
      <img src="https://github.com/user-attachments/assets/61ded279-7c6c-42e1-9293-450ab3864497" width="150">
      <img src="https://github.com/user-attachments/assets/8215fdd3-9627-4ade-bfa5-b0019c988569" width="150">
      <img src="https://github.com/user-attachments/assets/48df6c94-7501-44a1-9e59-d3bfc5b9e7ba" width="150">
      <img src="https://github.com/user-attachments/assets/18f47ca7-547d-4ba0-b3db-310d9dc29116" width="150">
      <img src="https://github.com/user-attachments/assets/5f79d49e-8fe9-4ef6-93be-920d1bade35f" width="150">
      <img src="https://github.com/user-attachments/assets/0d4b20e2-8e62-4c85-ab6c-2254415f5947" width="150">
      <img src="https://github.com/user-attachments/assets/f9fa0fda-5de6-4419-9a00-48969aaf6539" width="150">
</div>

## 🤝 Usage Acknowledgement
Notice we only authorize using the proposed dataset for scientific research. One should NOT use it for commercial purposes without our authorization.

## 🗓️ Schedule
**[2024.03.26]** Release [online demo](https://huggingface.co/spaces/posterllava/PosterLLaVA) and [pre-trained model](https://huggingface.co/posterllava/posterllava_v0) on hugging face🤗.

**[2024.06.05]** Release [arXiv](https://arxiv.org/abs/2406.02884) paper📝.

**[2024.07.04]** Release [QB-Poster](https://drive.google.com/file/d/1gRHTidpU0nePpjtDQElIVbAts8ziCkVh/view?usp=drive_link) dataset📊. (raw files contain <u>original poster images</u> and <u>JSON annotations</u>, inpainting and saliency detection techniques are needed for obtaining background images and saliency maps. Our paper used [lama](https://github.com/saic-mdal/lama) for inpainting and [basenet](https://github.com/xuebinqin/BASNet) for saliency detection.)

**[2024.07.04]** Release [User-Constrained](https://drive.google.com/file/d/1dlfxTC6QaV3Piyn655TMvTEv7-tCWuWk/view?usp=drive_link) dataset📊. (only include <u>user-constraint annotation</u> files. please refer to the [CGL-dataset](https://tianchi.aliyun.com/dataset/142692/notebook) and [PosterLayout](http://39.108.48.32/mipl/PosterLayout/) dataset to get the poster images and bounding box annotations.)

**[2024.07.04]** Release data pre-processing, training, and inferencing code.

**[2024.08.29]** An automatic text-to-poster system **PosterGen** (with *PosterLLaVA* as the backbone) will soon be open-sourced to supplement this work. 

**[Coming Soon]** Release evaluation code.

## Environment

Run the following code to build the environment.

```shell
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Data Processing 

Download the dataset files and arrange them as follows (QB-Poster as an example). Run the saliency detection method to get 'saliency_map' and the inpainting method to get 'inpainted_1x' and 'inpainted_1d5x' (used for inference and training respectively; notice we randomly inpainted 0.5x more regions besides the ground-truth bounding box area to avoid overfitting.)

```
├── data
│  ├── prompt_template.txt
│  └── qbposter <--
│      ├── get_prompt.py
|      └── raw
│          ├── original_poster
│          ├── saliency_map
│          ├── inpainted_1x
│          ├── inpainted_1d5x
│          └── annotation.json
...
└── README.md
```

Run the data preprocessing script.

```shell
python data/qbposter/get_prompt.py
```

Ultimately you will get two processed JSON files (each containing instruction-answer pairs) like this.

```
├── data
│  ├── prompt_template.txt
│  └── qbposter
│        ├── get_prompt.py
│        ├── qbposter_train_instruct.json <--
│        └── qbposter_val_instruct.json   <--
...
└── README.md
```

## Training
Please download [LLaVa-v1.5](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md#llava-v15) pre-trained checkpoint and [CLIP](https://huggingface.co/openai/clip-vit-large-patch14-336) vision encoder first and put it in the 'huggingface' subfolder.

```
├── data
├── huggingface <--
|      ├── llava-v1.5-7b
|      └── clip-vit-large-patch14-336
├── scripts
|      └── qbposter
|            ├── finetune.sh <--
|            └── inference.sh
...
└── README.md
```

Then run the following script.

```shell
qbposter/finetune.sh
```

## Inference
Please download the pre-trained [PosterLLaVa_v0](https://huggingface.co/posterllava/posterllava_v0) checkpoint, which is initialized with LLaVa-v1.5 checkpoint and fine-tuned on the following combined datasets.

- 7k banner layouts from *Ad Banner dataset*.
- 60k commercial poster layouts from *CGL-dataset* and *PosterLayout* with text constraints.
- 4k social media poster layouts from *QB-Poster* dataset.

Put it in the 'pretrained_model' subfolder.

```
├── data
├── huggingface
├── pretrained_model <--
|      └── posterllava_v0
├── scripts
|      └── qbposter
|            ├── finetune.sh
|            └── inference.sh <--
...
└── README.md
```

Then run the following script to generate JSON format layout.

```shell
qbposter/inference.sh
```

## Evaluation

Coming Soon...

## Citation

If you find this project/paper useful, please give us a star/citation.

```
@misc{yang2024posterllava,
      title={PosterLLaVa: Constructing a Unified Multi-modal Layout Generator with LLM}, 
      author={Tao Yang and Yingmin Luo and Zhongang Qi and Yang Wu and Ying Shan and Chang Wen Chen},
      year={2024},
      eprint={2406.02884},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.02884}, 
}
```
