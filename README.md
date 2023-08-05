# Zero-Shot Dialogue Relation Extraction by Relating Explainable Triggers and Relation Names

This repo is the implementation of [Zero-Shot Dialogue RE (Xu and Chen, 2023)](https://aclanthology.org/2023.nlp4convai-1.10.pdf).
![image](https://github.com/MiuLab/UnseenDRE/assets/2268109/e67142c4-1b7d-4351-960f-0f9230845f83)



## Training
```shell
bash run.sh
```

## Testing
```shell
bash predict.sh
```

## Reference
Please cite the following paper
```shell
@inproceedings{xu-chen-2023-zero,
    title = "Zero-Shot Dialogue Relation Extraction by Relating Explainable Triggers and Relation Names",
    author = "Xu, Ze-Song  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 5th Workshop on NLP for Conversational AI (NLP4ConvAI 2023)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.nlp4convai-1.10",
    pages = "123--128",
    abstract = "Developing dialogue relation extraction (DRE) systems often requires a large amount of labeled data, which can be costly and time-consuming to annotate. In order to improve scalability and support diverse, unseen relation extraction, this paper proposes a method for leveraging the ability to capture triggers and relate them to previously unseen relation names. Specifically, we introduce a model that enables zero-shot dialogue relation extraction by utilizing trigger-capturing capabilities. Our experiments on a benchmark DialogRE dataset demonstrate that the proposed model achieves significant improvements for both seen and unseen relations. Notably, this is the first attempt at zero-shot dialogue relation extraction using trigger-capturing capabilities, and our results suggest that this approach is effective for inferring previously unseen relation types. Overall, our findings highlight the potential for this method to enhance the scalability and practicality of DRE systems.",
}

```
