from package.dataset import DataLoader, NYT10TrainDataset, NYT10TestDataset
from package.utils import move_tensor_to_device, set_random_seed
from package.model import PCNNsConfig, PCNNs
from package.metrics import Score
from transformers import BertModel, BertTokenizerFast, BertConfig
from torch.optim import Adam
from pprint import pprint
from tqdm import tqdm

import torch

set_random_seed(42)

bag_size = 4
batch_size = 64
max_length = 512
accumulation_steps = 1
lr = 1e-3
epoch = 50
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
num_workers = 4

mode_name = 'bert-base-cased'
bert_tokenizer = BertTokenizerFast.from_pretrained(mode_name)
bert_config = BertConfig.from_pretrained(mode_name)
bert = BertModel.from_pretrained(mode_name)

for param in bert.parameters():
    param.requires_grad = False

dataset = NYT10TrainDataset('resource/nyt/train-reading-friendly.json', bert_tokenizer, bag_size, max_length)
dataloader_train = DataLoader(dataset, collate_fn=dataset.fast_collate_fn,
                              batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
dataset = NYT10TestDataset('resource/nyt/test-reading-friendly.json', bert_tokenizer, max_length)
dataloader_test = DataLoader(dataset, collate_fn=dataset.collate_fn,
                             batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

pcnns_config = PCNNsConfig(num_pieces=3, num_relations=dataset.num_relations, max_length=max_length,
                           position_embedding_dim=5, word_embedding_dim=bert_config.hidden_size,
                           kernel_sizes=[3], hidden_dims=[256], dropout_rate=0.5)
pcnns_model = PCNNs(bert, pcnns_config).to(device)

pcnns_model.reset_parameters()
optimizer = Adam(filter(lambda p: p.requires_grad, pcnns_model.parameters()), lr=lr)

if __name__ == '__main__':

    for _ in range(epoch):
        pcnns_model.train()
        with tqdm(desc='Training', total=len(dataloader_train)) as t:
            total_loss = None
            for i, bags in enumerate(dataloader_train):

                bags = move_tensor_to_device(bags, device)
                loss = pcnns_model.fast_multi_instance_learning_loss(bags)

                if total_loss:
                    total_loss = total_loss * 0.995 + loss * 0.005
                else:
                    total_loss = loss

                loss.backward()
                t.update(1)
                t.set_postfix(loss=float(total_loss))

                if (1 + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

        pcnns_model.eval()
        with torch.no_grad():

            score = Score()
            for i, batch in enumerate(tqdm(dataloader_test, desc='Testing')):
                batch = move_tensor_to_device(batch, device)

                logits = pcnns_model(batch['token_ids'], batch['pos_h_ids'], batch['pos_t_ids'], batch['mask'])
                y_pred = torch.argmax(logits, -1)
                y_true = batch['label']

                score.update(y_pred, y_true)

            pprint(score.compute())
