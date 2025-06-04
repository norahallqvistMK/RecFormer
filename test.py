import torch
from recformer import RecformerModel, RecformerConfig, RecformerForSeqRec

import os
import json
import torch
from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from functools import partial

from utils import read_json, AverageMeterSet, Ranker
from recformer import RecformerForSeqRec, RecformerTokenizer, RecformerConfig
from collator import EvalDataCollatorWithPadding
from dataloader import RecformerEvalDataset

# strict=False because RecformerForSeqRec doesn't have lm_head
def eval(model, dataloader, args):

    model.eval()

    ranker = Ranker(args.metric_ks)
    average_meter_set = AverageMeterSet()

    for batch, labels in tqdm(dataloader, ncols=100, desc='Evaluate'):

        for k, v in batch.items():
            batch[k] = v.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            scores = model(**batch)

        res = ranker(scores, labels)

        metrics = {}
        for i, k in enumerate(args.metric_ks):
            metrics["NDCG@%d" % k] = res[2*i]
            metrics["Recall@%d" % k] = res[2*i+1]
        metrics["MRR"] = res[-3]
        metrics["AUC"] = res[-2]

        for k, v in metrics.items():
            average_meter_set.update(k, v)

    average_metrics = average_meter_set.averages()

    return average_metrics

def load_data(args):

    train = read_json(os.path.join(args.data_path, args.train_file), True)
    val = read_json(os.path.join(args.data_path, args.dev_file), True)
    test = read_json(os.path.join(args.data_path, args.test_file), True)
    item_meta_dict = json.load(open(os.path.join(args.data_path, args.meta_file)))
    
    item2id = read_json(os.path.join(args.data_path, args.item2id_file))
    id2item = {v:k for k, v in item2id.items()}

    item_meta_dict_filted = dict()
    for k, v in item_meta_dict.items():
        if k in item2id:
            item_meta_dict_filted[k] = v

    return train, val, test, item_meta_dict_filted, item2id, id2item


tokenizer_glb = None  # define at global level

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
    # path and file
    parser.add_argument('--pretrain_ckpt', type=str, default='longformer_ckpt/recformer_seqrec_ckpt.bin')
    parser.add_argument('--data_path', type=str, default="finetune_data")
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--model_name_or_path', type=str, default='allenai/longformer-base-4096')
    parser.add_argument('--train_file', type=str, default='train.json')
    parser.add_argument('--dev_file', type=str, default='val.json')
    parser.add_argument('--test_file', type=str, default='test.json')
    parser.add_argument('--item2id_file', type=str, default='smap.json')
    parser.add_argument('--meta_file', type=str, default='meta_data.json')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--metric_ks', nargs='+', type=int, default=[10, 50], help='ks for Metric@k')

    # data process
    parser.add_argument('--preprocessing_num_workers', type=int, default=8, help="The number of processes to use for the preprocessing.")
    parser.add_argument('--dataloader_num_workers', type=int, default=0)

    # eval
    parser.add_argument('--device', type=int, default=0)

    args = parser.parse_args()
    print(args)

    train, val, test, item_meta_dict, item2id, id2item = load_data(args)

    config = RecformerConfig.from_pretrained(args.model_name_or_path)
    config.max_attr_num = 3  # max number of attributes for each item
    config.max_attr_length = 32 # max number of tokens for each attribute
    config.max_item_embeddings = 51 # max number of items in a sequence +1 for cls token
    config.attention_window = [64] * 12 # attention window for each layer
    config.item_num = len(item2id)
    tokenizer = RecformerTokenizer.from_pretrained(args.model_name_or_path, config)
    print(tokenizer.config)
    # model = RecformerModel(config)
    # model.load_state_dict(torch.load('longformer_ckpt/recformer_ckpt.bin'))

    model = RecformerForSeqRec(config)
    model.load_state_dict(torch.load(args.pretrain_ckpt), strict=False)
    
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
        pool = Pool(processes=args.preprocessing_num_workers, initializer=init_tokenizer, initargs=(args.model_name_or_path,))
        pool_func = pool.imap(func=_par_tokenize_doc, iterable=item_meta_dict.items())
        doc_tuples = list(tqdm(pool_func, total=len(item_meta_dict), ncols=100, desc=f'[Tokenize] {path_corpus}'))
        tokenized_items = {item2id[item_id]: [input_ids, token_type_ids] for item_id, input_ids, token_type_ids in doc_tuples}
        pool.close()
        pool.join()

        torch.save(tokenized_items, path_tokenized_items)

    tokenized_items = torch.load(path_tokenized_items)
    print(f'Successfully load {len(tokenized_items)} tokenized items.')

    eval_data_collator = EvalDataCollatorWithPadding(tokenizer, tokenized_items)

    test_data = RecformerEvalDataset(train, val, test, mode='test', collator=eval_data_collator)

    test_loader = DataLoader(test_data, 
                            batch_size=args.batch_size, 
                            collate_fn=test_data.collate_fn)
    
    args = parser.parse_args()
    train, val, test, item_meta_dict, item2id, id2item = load_data(args)

    test_metrics = eval(model, test_loader, args)
    print(f'Test set: {test_metrics}')

if __name__ == "__main__":
    main()