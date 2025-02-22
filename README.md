# Learning Fine-Grained Representations through Textual Token Disentanglement in Composed Video Retrieval

This repo is the official implementation of ICLR 2025 paper "Learning Fine-Grained Representations through Textual Token Disentanglement in Composed Video Retrieval".
By Yue Wu, Zhaobo Qi, Yiling Wu, Junshu Sun, Yaowei Wang, Shuhui Wang

# Introduction

<p>With the explosive growth of video data, finding videos that meet detailed requirements in large datasets has become a challenge. To address this, the composed video retrieval task has been introduced, enabling users to retrieve videos using complex queries that involve both visual and textual information. However, the inherent heterogeneity between modalities poses significant challenges. Textual data is highly abstract, while video content contains substantial redundancy. This modality gap in information representation makes existing methods struggle with the fine-grained fusion and alignment required for fine-grained composed retrieval. To overcome these challenges, we introduce FineCVR-1M, a fine-grained composed video retrieval dataset containing 1,010,071 video-text triplets with detailed textual descriptions. This dataset is constructed through an automated process that identifies key concept changes between video pairs to generate textual descriptions for both static and action concepts. For fine-grained retrieval methods, the key challenge lies in understanding the detailed requirements. Text descriptions serve as clear expressions of intent, allowing models to distinguish fine-grained needs through textual feature disentanglement. Therefore, we propose a textual Feature Disentanglement and Cross-modal Alignment framework FDCA that disentangles features at both the sentence and token levels. At the sequence level, we separate text features into retained and injected features. At the token level, an Auxiliary Token Disentangling mechanism is proposed to disentangle texts into retained, injected, and excluded tokens. The disentanglement at both levels extracts fine-grained features, which are aligned and fused with reference video to extract global representations for video retrieval. Experiments on FineCVR-1M dataset demonstrate the superior performance of FDCA.</p>

# Cite

If you use our dataset or method in your research, please cite our paper:
```
 @article{yue25finecvr,
        title = {Learning Fine-Grained Representations through Textual Token Disentanglement in Composed Video Retrieval},
        author = {Yue Wu, Zhaobo Qi, Yiling Wu, Junshu Sun, Yaowei Wang, Shuhui Wang},
        journal = {ICLR},
        year = {2025}
    }
```