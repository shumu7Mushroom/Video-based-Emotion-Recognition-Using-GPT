# Video-based-Emotion-Recognition-Using-GPT
Generate emotion key words by sending video clips to ChatGPT

# Description

In this project i use the work of [keyframe-extraction-for-video-summarization](https://github.com/ttharden/Keyframe-Extraction-for-video-summarization) to pick up keyframes.
First, The large model TransNetV2 was utilized to conduct shot segmentations, and the large model CLIP was employed to extract semantic features for each frame within each shot. Second, an adaptive clustering method is devised to automatically determine the optimal clusters, based on which we performed candidate keyframe selection and redundancy elimination shot by shot. Finally, a keyframe set was obtained by concatenating keyframes of all shots in chronological order.

After that, call gpt4-turbo to recognize the emotion of video by sending him video clips. Other models should also work.

# Environment setting

First, conda create environment
```bash
conda env create --name keyframe python==3.10
conda activate keyframe
```

Then pull the project and download requirements
```bash
git clone https://github.com/shumu7Mushroom/Video-based-Emotion-Recognition-Using-GPT.git
cd Video-based-Emotion-Recognition-Using-GPT
pip install -r requirements.txt
```

By the way, if you havn't download ffmpeg in your computer, you can [download](https://www.ffmpeg.org/) it and add the path to your system's environment variables

# Use it 

_Step 1:_ Run pipeline.py to get the keyframe sequence


_Step 2:_ Run save.py to save the keyframes

_Step 3:_ Run Callgpt.py to call gpt to get emotion


# Acknowlegement

I refer the codes of following project

[CLIP](https://github.com/openai/CLIP)

[TransnetV2](https://github.com/soCzech/TransNetV2)

[keyframe-extraction-for-video-summarization](https://github.com/ttharden/Keyframe-Extraction-for-video-summarization)

# Reference
Tomas Soucek and Jakub Lokoc, “Transnet V2: an effective deep network architecture for fast shot transition detection,” arXiv:2008.04838, pp. 1–4, 2020.

Yang, An and Pan, Junshu and Lin, Junyang and Men, Rui and Zhang, Yichang and Zhou, Jingren and Zhou, Chang, "Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese" arXiv preprint arXiv:2211.01335,2022

Sandra Eliza Fontes de Avila, Ana Paula Brandao Lopes, Antonio da Luz Jr., and Arnaldo de Albuquerque Araujo, “VSUMM: A mechanism designed to produce static video summaries and a novel evaluation method,” Pattern Recognit. Lett., vol. 32, no. 1, pp. 56–68, 2011.

Hana Gharbi, Sahbi Bahroun, Mohamed Massaoudi, and Ezzeddine Zagrouba, “Key frames extraction using graph modularity clustering for efficient video summarization,” in ICASSP, 2017, pp. 1502–1506.

H.M. Nandini, H.K. Chethan, and B.S. Rashmi, “Shot based keyframe extraction using edge-lbp approach,” Journal of King Saud University - Computer and Information Sciences, vol. 34, no. 7, pp. 4537–4545, 2022.

Luis Carlos Garcia-Peraza, Sebastien Ourselin, and Tom Vercauteren, VideoSum: A Python Library for Surgical Video Summarization, pp. 1–2, 2023.











