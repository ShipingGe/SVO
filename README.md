# SVO
Official Repository for the SIGIR 2024 paper: Short Video Ordering via Position Decoding and Successor Prediction.

The code and the dataset will be released soon after the paper is published officially.

2024.7.15: The code has been released.
2024.12.18: The SVO dataset has been released.

## About
**Short video collection** is an easy way for users to consume coherent content on various online short video platforms, such as TikTok, YouTube, Douyin, and WeChat Channel. 
However, short video creators occasionally publish videos in a disorganized manner due to various reasons, such as revisions, secondary creations, deletions, and reissues, which often result in a poor browsing experience for users. 
Therefore, accurately reordering videos within a collection based on their content coherence is a vital task that can enhance user experience and presents an intriguing research problem in the field of video narrative reasoning. 
In this work, we curate a dedicated multimodal dataset for this Short Video Ordering (SVO) task and present the performance of some benchmark methods on the dataset. 
In addition, we further propose an advanced SVO framework with the aid of position decoding and successor prediction. Extensive experiments demonstrate that our method achieves the best performance on our open SVO dataset, and each component of the framework contributes to the final performance. 

<div align="center">
    <img src="Figure/svo.png" width="60%" alt="svo" align="center">
</div>

## Dataset
The data required to re-implement our work has been uploaded to the following URL:
https://drive.google.com/drive/folders/11FcKWZBlPoBOMVMQOcSqD0DuisZbkgD7?usp=sharing

**Important Note:**
This dataset is intended for research purposes only and may not be used for commercial purposes.
Due to considerations regarding copyright and the protection of user privacy, we are unable to directly open-source the original video data. 
Therefore, we have made available the video frame features extracted using the CLIP model. 
This data can not be used technically, nor should anyone attempt to use it, to reconstruct the original videos. 
However, it is sufficient for reproducing our methods or for further research that does not infringe upon user privacy.

## License
<p xmlns:cc="http://creativecommons.org/ns#" ><a rel="cc:attributionURL" href="https://github.com/ShipingGe/SVO">This work</a> is licensed under <a href="https://creativecommons.org/licenses/by-nc/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC 4.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1" alt=""></a></p>

## Citation
If you find our paper and code useful in your research, please consider giving a star ⭐ and citation 📝 :)

```
@inproceedings{ge2024short,
  title={Short Video Ordering via Position Decoding and Successor Prediction},
  author={Ge, Shiping and Chen, Qiang and Jiang, Zhiwei and Yin, Yafeng and Chen, Ziyao and Gu, Qing},
  booktitle={Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={2167--2176},
  year={2024}
}
```
