# [Large Language Models for Propaganda Span Annotation](https://aclanthology.org/2024.findings-emnlp.850.pdf)

![License](https://img.shields.io/badge/license-CC--BY--NC--SA-blue) [![Paper](https://img.shields.io/badge/Paper-Download%20PDF-green)](https://aclanthology.org/2024.findings-emnlp.850.pdf)

Our study investigates whether large language models (LLMs), such as GPT-4, can effectively extract propagandistic spans. We further study the potential of employing the model to collect more cost-effective annotations. Finally, we examine the effectiveness of labels provided by GPT-4 in training smaller language models for the task.
In this repo we release full human annotations, consolidated gold labels, and annotations provided by GPT-4 in different annotator roles.

## Overview
<p align="center">
<picture>
<img alt = "Existing span-level annotation process requiring human annotators and expert consolidators, while our proposed solution uses GPT-4 to support annotation and consolidation." src="https://github.com/user-attachments/assets/2744ef4d-3ec4-4939-97d3-5c188a100075", width="460" height="400"/>
</picture>
</p>


**Our repo provides the following scripts:**
- chatgpt4_prop_detection_annot.py: a script to run GPT-4 as an annotator in different annotation roles.
- span_detect_eval.py: evaluation script computing modified F1 score between gold and predicted propaganda spans.

**To run chatgpt4_prop_detection_annot.py:**
```bash
python scripts/chatgpt4_prop_detection_annot.py --input_file annotations/human/ArMPro_span_train_full-annotations.jsonl --output_file gpt4_predictions.jsonl --err_output_file error_cases.jsonl --role annot --env gpt4_keys.env
```

#### Parameters

- `--input_file` input human annotations file.
- `--output_file` output file for model annotations.
- `--err_output_file`output file for failed model responses.
- `--role` LLM annotation role. Possible options: annot | select | cons .
- `--env` API key file.

**To run span_detect_eval.py:**
```bash
python scripts/span_detect_eval.py --gold_file annotations/human/ArMPro_span_train.jsonl --pred_file gpt4_predictions.jsonl
```

#### Parameters

- `--gold_file` gold annotations file.
- `--pred_file` model annotations file.



## Citation

Please cite our paper when referring to this work:

- Maram Hasanain, Fatema Ahmad, and Firoj Alam. 2024. Large Language Models for Propaganda Span Annotation. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 14522–14532, Miami, Florida, USA. Association for Computational Linguistics.


```
@inproceedings{hasanain-etal-2024-large,
    title = "Large Language Models for Propaganda Span Annotation",
    author = "Hasanain, Maram  and
      Ahmad, Fatema  and
      Alam, Firoj",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.850",
    pages = "14522--14532",
    abstract = "The use of propagandistic techniques in online content has increased in recent years aiming to manipulate online audiences. Fine-grained propaganda detection and extraction of textual spans where propaganda techniques are used, are essential for more informed content consumption. Automatic systems targeting the task over lower resourced languages are limited, usually obstructed by lack of large scale training datasets. Our study investigates whether Large Language Models (LLMs), such as GPT-4, can effectively extract propagandistic spans. We further study the potential of employing the model to collect more cost-effective annotations. Finally, we examine the effectiveness of labels provided by GPT-4 in training smaller language models for the task. The experiments are performed over a large-scale in-house manually annotated dataset. The results suggest that providing more annotation context to GPT-4 within prompts improves its performance compared to human annotators. Moreover, when serving as an expert annotator (consolidator), the model provides labels that have higher agreement with expert annotators, and lead to specialized models that achieve state-of-the-art over an unseen Arabic testing set. Finally, our work is the first to show the potential of utilizing LLMs to develop annotated datasets for propagandistic spans detection task prompting it with annotations from human annotators with limited expertise. All scripts and annotations will be shared with the community.",
}
```
