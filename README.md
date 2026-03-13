# Value Alignment in LLMs
<!-- # value_action_data -->

This repository replicates the paper [Mind the Value-Action Gap: Do LLMs Act in Alignment with Their Values?](https://arxiv.org/pdf/2501.15463?) from [EMNLP 2025](https://2025.emnlp.org/).

This paper aims to answer *To what extent do LLM-generated value statements align with their value-informed actions?* (Value Alignment between LLM's **value claim & the corresponding actions**,

We replicate the original authors Value-Action frameworks using their datasets for prompting models and their file eval_alignment_llama3.py for evaluating Llama's performance on the given tests. 

Original paper citation: 
```bibtex
@article{shen2025mind,
  title={Mind the Value-Action Gap: Do LLMs Act in Alignment with Their Values?},
  author={Shen, Hua and Clark, Nicholas and Mitra, Tanushree},
  journal={arXiv preprint arXiv:2501.15463},
  year={2025}
}
