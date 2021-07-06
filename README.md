# PCNNs
Bert + PCNNs (-One) for RE Task

## Requirements
- python 3.8
- pytorch 1.8.1
- transformers 4.1.1
- scikit-learn 0.24.2
- matplotlib 3.3.4
- tqdm

## Run
```shell
python main.py
```

## configure what you want
```python
set_random_seed(42)
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
num_workers = 4

bag_size = 4
batch_size = 64
max_length = 512
accumulation_steps = 1
lr = 1e-3
epoch = 50

mode_name = 'bert-base-cased'
bert_tokenizer = BertTokenizerFast.from_pretrained(mode_name)
bert_config = BertConfig.from_pretrained(mode_name)

dataset = NYT10TrainDataset('resource/nyt/train-reading-friendly.json', bert_tokenizer, bag_size, max_length)
dataloader_train = DataLoader(dataset, collate_fn=dataset.fast_collate_fn,
                              batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
dataset = NYT10TestDataset('resource/nyt/test-reading-friendly.json', bert_tokenizer, max_length)
dataloader_test = DataLoader(dataset, collate_fn=dataset.collate_fn,
                             batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

pcnns_config = PCNNsConfig(num_pieces=3, num_relations=dataset.num_relations, max_length=max_length,
                           position_embedding_dim=5, word_embedding_dim=bert_config.hidden_size,
                           kernel_sizes=[3], hidden_dims=[256], dropout_rate=0.5)
```

## Reference
- Distant Supervision for Relation Extraction via Piecewise Convolutional Neural Networks. EMNLP. 2015. [[paper]](https://doi.org/10.18653/v1/d15-1203)
- OpenNRE [[github]](https://github.com/thunlp/OpenNRE)
- pytorch-relation-extraction [[github]](https://github.com/ShomyLiu/pytorch-relation-extraction)