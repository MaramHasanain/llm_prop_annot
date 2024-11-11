# Large Language Models for Propaganda Span Annotation

Our study investigates whether large language models (LLMs), such as GPT-4, can effectively extract propagandistic spans. We further study the potential of employing the model to collect more cost-effective annotations. Finally, we examine the effectiveness of labels provided by GPT-4 in training smaller language models for the task.
In this repo we release full human annotations, consolidated gold labels, and annotations provided by GPT-4 in different annotator roles.

## Overview
<p align="center">
<picture>
<img alt = "Existing span-level annotation process requiring human annotators and expert consolidators, while our proposed solution uses GPT-4 to support annotation and consolidation." src="" width="510" height="160"/>
</picture>
</p>


Our repo provides the following scripts:
- chatgpt4_prop_detection_annot.py: a script to run GPT-4 as an annotator in different annotation roles.
- span_detect_eval.py: evaluation script computing modified F1 score between gold and predicted propaganda spans. 

To run chatgpt4_prop_detection_annot.py:
```bash
python scripts/chatgpt4_prop_detection_annot.py --input_file annotations/human/ArMPro_span_train_full-annotations.jsonl --output_file gpt4_predictions.jsonl --err_output_file error_cases.jsonl --role annot --env gpt4_keys.env
```

#### Parameters

- `--input_file` input human annotations file.
- `--output_file` output file for model annotations.
- `--err_output_file`output file for failed model responses.
- `--role` LLM annotation role. Possible options: annot | select | cons .
- `--env` API key file.

To run span_detect_eval.py:
```bash
python scripts/span_detect_eval.py --gold_file annotations/human/ArMPro_span_train.jsonl --pred_file gpt4_predictions.jsonl
```

#### Parameters

- `--gold_file` gold annotations file.
- `--pred_file` model annotations file.



## Citation

Please cite our paper when referring to this work:

```
@inproceedings{hasanain2023large,
  title={Large language models for propaganda span annotation},
  author={Hasanain, Maram and Ahmad, Fatema and Alam, Firoj},
  booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
  year={2024}
}
```