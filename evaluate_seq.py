import os
import json
import torch
from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from functools import partial

from utils import read_json, AverageMeterSet, Ranker
from recformer import RecformerForSeqRec, RecformerTokenizer, RecformerConfig, RecformerModel
from collator import EvalDataCollatorWithPadding
from dataloader import RecformerEvalDataset


def encode_all_items(model: RecformerModel, tokenizer: RecformerTokenizer, tokenized_items, args):
    model.eval()
    items = [v for k, v in sorted(tokenized_items.items())]
    item_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(items), args.batch_size), ncols=100, desc='Encode all items'):
            item_batch = [[item] for item in items[i:i+args.batch_size]]
            inputs = tokenizer.batch_encode(item_batch, encode_item=False)
            inputs = {k: torch.LongTensor(v).to(args.device) for k, v in inputs.items()}
            outputs = model(**inputs)
            item_embeddings.append(outputs.pooler_output)

    return torch.cat(item_embeddings, dim=0)


def eval(model, dataloader, args):
    model.eval()
    ranker = Ranker(args.metric_ks)
    average_meter_set = AverageMeterSet()

    for batch, labels in tqdm(dataloader, ncols=100, desc='Evaluate'):
        batch = {k: v.to(args.device) for k, v in batch.items()}
        labels = labels.to(args.device)

        with torch.no_grad():
            scores = model(**batch)

        res = ranker(scores, labels)
        metrics = {
            f"NDCG@{k}": res[2 * i]
            for i, k in enumerate(args.metric_ks)
        }
        metrics.update({
            f"Recall@{k}": res[2 * i + 1]
            for i, k in enumerate(args.metric_ks)
        })
        metrics["MRR"] = res[-3]
        metrics["AUC"] = res[-2]

        for k, v in metrics.items():
            average_meter_set.update(k, v)

    return average_meter_set.averages()


def load_data(args):
    train = read_json(os.path.join(args.data_path, args.train_file), True)
    val = read_json(os.path.join(args.data_path, args.dev_file), True)
    test = read_json(os.path.join(args.data_path, args.test_file), True)
    item_meta_dict = json.load(open(os.path.join(args.data_path, args.meta_file)))
    item2id = read_json(os.path.join(args.data_path, args.item2id_file))
    item_meta_dict = {k: v for k, v in item_meta_dict.items() if k in item2id}
    id2item = {v: k for k, v in item2id.items()}
    return train, val, test, item_meta_dict, item2id, id2item


tokenizer_glb = None


def init_tokenizer(model_name_or_path):
    global tokenizer_glb
    config = RecformerConfig.from_pretrained(model_name_or_path)
    config.max_attr_num = 3
    config.max_attr_length = 32
    config.max_item_embeddings = 51
    config.attention_window = [64] * 12
    tokenizer_glb = RecformerTokenizer.from_pretrained(model_name_or_path, config)


def _par_tokenize_doc(doc):
    global tokenizer_glb
    item_id, item_attr = doc
    input_ids, token_type_ids = tokenizer_glb.encode_item(item_attr)
    return item_id, input_ids, token_type_ids


def main():
    parser = ArgumentParser()
    parser.add_argument('--pretrain_ckpt', type=str, default='longformer_ckpt/recformer_seqrec_ckpt.bin')
    parser.add_argument('--data_path', type=str, default="finetune_data")
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--model_name_or_path', type=str, default='allenai/longformer-base-4096')
    parser.add_argument('--train_file', type=str, default='train.json')
    parser.add_argument('--dev_file', type=str, default='val.json')
    parser.add_argument('--test_file', type=str, default='test.json')
    parser.add_argument('--item2id_file', type=str, default='smap.json')
    parser.add_argument('--meta_file', type=str, default='meta_data.json')
    parser.add_argument('--preprocessing_num_workers', type=int, default=8)
    parser.add_argument('--dataloader_num_workers', type=int, default=0)
    parser.add_argument('--num_train_epochs', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--finetune_negative_sample_size', type=int, default=1000)
    parser.add_argument('--metric_ks', nargs='+', type=int, default=[10, 50])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fix_word_embedding', action='store_true')
    parser.add_argument('--verbose', type=int, default=3)

    args = parser.parse_args()
    args.device = torch.device(f'cuda:{args.device}' if args.device >= 0 else 'cpu')
    print(args)

    train, val, test, item_meta_dict, item2id, id2item = load_data(args)

    config = RecformerConfig.from_pretrained(args.model_name_or_path)
    config.max_attr_num = 3
    config.max_attr_length = 32
    config.max_item_embeddings = 51
    config.attention_window = [64] * 12
    config.item_num = len(item2id)

    tokenizer = RecformerTokenizer.from_pretrained(args.model_name_or_path, config)
    model = RecformerForSeqRec(config)
    model.load_state_dict(torch.load(args.pretrain_ckpt), strict=False)
    model.to(args.device)

    global tokenizer_glb
    tokenizer_glb = tokenizer

    path_corpus = Path(args.data_path)
    dir_preprocess = path_corpus / 'preprocess'
    dir_preprocess.mkdir(exist_ok=True)

    path_output = Path(args.output_dir) / path_corpus.name
    path_output.mkdir(exist_ok=True, parents=True)

    path_tokenized_items = dir_preprocess / f'tokenized_items_{path_corpus.name}'
    if path_tokenized_items.exists():
        print(f'[Preprocessor] Use cache: {path_tokenized_items}')
    else:
        print(f'Loading attribute data {path_corpus}')
        with Pool(processes=args.preprocessing_num_workers, initializer=init_tokenizer, initargs=(args.model_name_or_path,)) as pool:
            pool_func = pool.imap(func=_par_tokenize_doc, iterable=item_meta_dict.items())
            doc_tuples = list(tqdm(pool_func, total=len(item_meta_dict), ncols=100, desc=f'[Tokenize] {path_corpus}'))
            tokenized_items = {item2id[item_id]: [input_ids, token_type_ids] for item_id, input_ids, token_type_ids in doc_tuples}
        torch.save(tokenized_items, path_tokenized_items)

    tokenized_items = torch.load(path_tokenized_items)
    print(f'Successfully loaded {len(tokenized_items)} tokenized items.')

    eval_data_collator = EvalDataCollatorWithPadding(tokenizer, tokenized_items)
    test_data = RecformerEvalDataset(train, val, test, mode='test', collator=eval_data_collator)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=test_data.collate_fn)

    if args.fix_word_embedding:
        print('Fix word embeddings.')
        for param in model.longformer.embeddings.word_embeddings.parameters():
            param.requires_grad = False

    path_item_embeddings = dir_preprocess / f'item_embeddings_{path_corpus.name}'
    if path_item_embeddings.exists():
        print(f'[Item Embeddings] Use cache: {path_item_embeddings}')
    else:
        print('Encoding items.')
        item_embeddings = encode_all_items(model.longformer, tokenizer, tokenized_items, args)
        torch.save(item_embeddings, path_item_embeddings)

    item_embeddings = torch.load(path_item_embeddings)
    model.init_item_embedding(item_embeddings)
    model.to(args.device)

    test_metrics = eval(model, test_loader, args)
    print(f'Test set: {test_metrics}')


if __name__ == "__main__":
    main()
