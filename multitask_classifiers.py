import random
import numpy as np
import argparse
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import BertModel, AdamW
from tqdm import tqdm

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)

from evaluation import model_eval_multitask, model_eval_test_multitask

TQDM_DISABLE = False

# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5

class MultitaskBERT(nn.Module):
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
        for param in self.bert.parameters():
            param.requires_grad = config.fine_tune_mode == 'full-model'
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifiers = nn.ModuleList([torch.nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES) for _ in range(config.num_layers)])
        self.classifier_paraphrase = torch.nn.Linear(BERT_HIDDEN_SIZE * 2, 1)
        self.classifier_sts = torch.nn.Linear(BERT_HIDDEN_SIZE * 2, 1)
        self.lte = nn.Linear(BERT_HIDDEN_SIZE, 1)  # Learning-to-Exit module

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # Get all hidden states from each transformer layer
        cls_states = [hidden_state[:, 0] for hidden_state in hidden_states]  # Take the [CLS] token's hidden state
        return cls_states

    def predict_sentiment(self, input_ids, attention_mask, exit_layer=None):
        hidden_states = self.forward(input_ids, attention_mask)
        if exit_layer is None:
            exit_layer = len(hidden_states) - 2  # Adjust to use the last classifier (not the embedding layer)
        logits = self.classifiers[exit_layer - 1](hidden_states[exit_layer])  # Adjust index to skip embedding layer
        return logits

    def predict_paraphrase(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        c1 = self.forward(input_ids_1, attention_mask_1)
        c2 = self.forward(input_ids_2, attention_mask_2)
        combined = torch.cat((c1[-1], c2[-1]), dim=1)
        res = self.dropout(combined)
        return self.classifier_paraphrase(res)

    def predict_similarity(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        c1 = self.forward(input_ids_1, attention_mask_1)
        c2 = self.forward(input_ids_2, attention_mask_2)
        combined = torch.cat((c1[-1], c2[-1]), dim=1)
        res = self.dropout(combined)
        return self.classifier_sts(res)

    def learning_to_exit(self, hidden_state):
        return torch.sigmoid(self.lte(hidden_state)).mean()  # Return the mean certainty as a scalar value

def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }
    torch.save(save_info, filepath)
    print(f"Saved the model to {filepath}")

def train_multitask_berxit(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    sst_train_data, num_labels, para_train_data, sts_train_data = load_multitask_data(args.sst_train, args.para_train, args.sts_train, split='train')
    sst_dev_data, num_labels, para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev, args.para_dev, args.sts_dev, split='dev')

    sst_train_dataset = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_dataset = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=sst_train_dataset.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=sst_dev_dataset.collate_fn)

    para_train_dataset = SentencePairDataset(para_train_data, args)
    para_dev_dataset = SentencePairDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(para_train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=para_train_dataset.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=para_dev_dataset.collate_fn)

    sts_train_dataset = SentencePairDataset(sts_train_data, args)
    sts_dev_dataset = SentencePairDataset(sts_dev_data, args, isRegression=True)

    sts_train_dataloader = DataLoader(sts_train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=sts_train_dataset.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=sts_dev_dataset.collate_fn)

    config = {'hidden_dropout_prob': args.hidden_dropout_prob, 'num_labels': num_labels, 'hidden_size': 768, 'data_dir': '.', 'fine_tune_mode': args.fine_tune_mode, 'num_layers': 12}
    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    # Alternating fine-tuning strategy
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        for batch in tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_mask, b_labels = (batch['token_ids'], batch['attention_mask'], batch['labels'])
            b_ids, b_mask, b_labels = b_ids.to(device), b_mask.to(device), b_labels.to(device)

            optimizer.zero_grad()
            exit_layer = epoch % len(model.classifiers)  # Ensure exit_layer is within range of classifiers
            logits = model.predict_sentiment(b_ids, b_mask, exit_layer=exit_layer)
            loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            (b_ids1, b_mask1, b_ids2, b_mask2, b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'], batch['token_ids_2'], batch['attention_mask_2'], batch['labels'], batch['sent_ids'])
            b_ids1, b_mask1, b_ids2, b_mask2, b_labels = b_ids1.to(device), b_mask1.to(device), b_ids2.to(device), b_mask2.to(device), b_labels.clone().detach().to(device, dtype=torch.float32)

            optimizer.zero_grad()
            logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
            loss = F.binary_cross_entropy_with_logits(logits.squeeze(), b_labels, reduction='sum') / args.batch_size
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        for batch in tqdm(sts_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            (b_ids1, b_mask1, b_ids2, b_mask2, b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'], batch['token_ids_2'], batch['attention_mask_2'], batch['labels'], batch['sent_ids'])
            b_ids1, b_mask1, b_ids2, b_mask2, b_labels = b_ids1.to(device), b_mask1.to(device), b_ids2.to(device), b_mask2.to(device), b_labels.clone().detach().to(device, dtype=torch.float32)

            optimizer.zero_grad()
            logits = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
            loss = F.mse_loss(logits.squeeze(), b_labels, reduction='sum') / args.batch_size
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss /= num_batches
        train_acc, train_f1, *_ = model_eval_multitask(sst_train_dataloader, para_train_dataloader, sts_train_dataloader, model, device)
        dev_acc, dev_f1, *_ = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss:.3f}, train acc :: {train_acc:.3f}, dev acc :: {dev_acc:.3f}")

        # Early exiting with learning-to-exit (LTE) method
        for batch in tqdm(sst_dev_dataloader, desc=f'validate-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_mask, b_labels = (batch['token_ids'], batch['attention_mask'], batch['labels'])
            b_ids, b_mask, b_labels = b_ids.to(device), b_mask.to(device), b_labels.to(device)
            hidden_states = model.forward(b_ids, b_mask)
            for i, hidden_state in enumerate(hidden_states):
                certainty = model.learning_to_exit(hidden_state)
                if certainty.item() > 0.9:  # Exit early if certainty is high
                    logits = model.classifiers[i](hidden_state)
                    break

def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels, para_test_data, sts_test_data = load_multitask_data(args.sst_test, args.para_test, args.sts_test, split='test')
        sst_dev_data, num_labels, para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev, args.para_dev, args.sts_dev, split='dev')

        sst_test_dataset = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_dataset = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=sst_test_dataset.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=sst_dev_dataset.collate_fn)

        para_test_dataset = SentencePairTestDataset(para_test_data, args)
        para_dev_dataset = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=para_test_dataset.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=para_dev_dataset.collate_fn)

        sts_test_dataset = SentencePairTestDataset(sts_test_data, args)
        sts_dev_dataset = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=sts_test_dataset.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=sts_dev_dataset.collate_fn)

        dev_sentiment_accuracy, dev_sst_y_pred, dev_sst_sent_ids, dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)

        test_sst_y_pred, test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = model_eval_test_multitask(sst_test_dataloader, para_test_dataloader, sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy:.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy:.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr:.3f}")
            f.write(f"id \t Predicted_Similarity \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similarity \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--fine-tune-mode", type=str, choices=('last-linear-layer', 'full-model'), default="full-model")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1.3e-5)
    parser.add_argument("--epochs", type=int, default=5)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.fine_tune_mode}-{args.epochs}-{args.lr}-multitask.pt'
    seed_everything(args.seed)
    train_multitask_berxit(args)
    test_multitask(args)
