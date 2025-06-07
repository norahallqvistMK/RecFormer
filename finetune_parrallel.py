import os
import json
import torch
from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import torch.nn as nn

from pytorch_lightning import seed_everything

from utils import read_json, AverageMeterSet, Ranker
from optimization import create_optimizer_and_scheduler
from recformer import RecformerModel, RecformerForSeqRec, RecformerTokenizer, RecformerConfig
from collator import FinetuneDataCollatorWithPadding, EvalDataCollatorWithPadding
from dataloader import RecformerTrainDataset, RecformerEvalDataset

def get_base_model(model):
    """Helper function to get the base model (unwrapped from DataParallel)"""
    return model.module if isinstance(model, nn.DataParallel) else model

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


def encode_all_items_multi_gpu(model: RecformerModel, tokenizer: RecformerTokenizer, tokenized_items, args):
    """
    Encode items using multiple GPUs to prevent CUDA OOM
    """
    model.eval()
    
    # Get available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs for item encoding")
    
    if num_gpus <= 1:
        # Fallback to single GPU if only one available
        return encode_all_items_single_gpu(model, tokenizer, tokenized_items, args)
    
    # Wrap model with DataParallel for multi-GPU inference
    model.to(args.device)
    if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
    
    items = sorted(list(tokenized_items.items()), key=lambda x: x[0])
    items = [ele[1] for ele in items]
    
    item_embeddings = []
    
    # Adjust batch size for multi-GPU (multiply by number of GPUs)
    effective_batch_size = args.batch_size * num_gpus
    
    with torch.no_grad():
        for i in tqdm(range(0, len(items), effective_batch_size), ncols=100, desc=f'Encode all items (Multi-GPU: {num_gpus})'):
            
            item_batch = [[item] for item in items[i:i+effective_batch_size]]
            
            if len(item_batch) == 0:
                continue
                
            inputs = tokenizer.batch_encode(item_batch, encode_item=False)
            
            for k, v in inputs.items():
                inputs[k] = torch.LongTensor(v).to(args.device)
            
            try:
                outputs = model(**inputs)
                item_embeddings.append(outputs.pooler_output.detach().cpu())  # Move to CPU immediately
                
                # Clear GPU cache periodically
                if i % (effective_batch_size * 10) == 0:
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"CUDA OOM at batch {i}, reducing batch size and retrying...")
                    torch.cuda.empty_cache()
                    
                    # Process this batch with smaller chunks
                    chunk_size = max(1, len(item_batch) // (num_gpus * 2))
                    for j in range(0, len(item_batch), chunk_size):
                        chunk = item_batch[j:j+chunk_size]
                        if len(chunk) == 0:
                            continue
                            
                        chunk_inputs = tokenizer.batch_encode(chunk, encode_item=False)
                        for k, v in chunk_inputs.items():
                            chunk_inputs[k] = torch.LongTensor(v).to(args.device)
                        
                        chunk_outputs = model(**chunk_inputs)
                        item_embeddings.append(chunk_outputs.pooler_output.detach().cpu())
                        
                    torch.cuda.empty_cache()
                else:
                    raise e
    
    # Concatenate all embeddings and move back to original device
    item_embeddings = torch.cat(item_embeddings, dim=0).to(args.device)
    
    return item_embeddings


def encode_all_items_single_gpu(model: RecformerModel, tokenizer: RecformerTokenizer, tokenized_items, args):
    """
    Original single GPU encoding with memory optimization
    """
    model.eval()

    items = sorted(list(tokenized_items.items()), key=lambda x: x[0])
    items = [ele[1] for ele in items]

    item_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(items), args.batch_size), ncols=100, desc='Encode all items (Single GPU)'):

            item_batch = [[item] for item in items[i:i+args.batch_size]]

            inputs = tokenizer.batch_encode(item_batch, encode_item=False)

            for k, v in inputs.items():
                inputs[k] = torch.LongTensor(v).to(args.device)

            try:
                outputs = model(**inputs)
                item_embeddings.append(outputs.pooler_output.detach().cpu())  # Move to CPU immediately
                
                # Clear cache every 50 batches
                if i % (args.batch_size * 50) == 0:
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"CUDA OOM at batch {i}, clearing cache and retrying with smaller batch...")
                    torch.cuda.empty_cache()
                    
                    # Retry with half batch size
                    half_batch_size = max(1, len(item_batch) // 2)
                    for j in range(0, len(item_batch), half_batch_size):
                        mini_batch = item_batch[j:j+half_batch_size]
                        if len(mini_batch) == 0:
                            continue
                            
                        mini_inputs = tokenizer.batch_encode(mini_batch, encode_item=False)
                        for k, v in mini_inputs.items():
                            mini_inputs[k] = torch.LongTensor(v).to(args.device)
                        
                        mini_outputs = model(**mini_inputs)
                        item_embeddings.append(mini_outputs.pooler_output.detach().cpu())
                        
                    torch.cuda.empty_cache()
                else:
                    raise e

    # Concatenate and move back to device
    item_embeddings = torch.cat(item_embeddings, dim=0).to(args.device)

    return item_embeddings


def encode_all_items(model: RecformerModel, tokenizer: RecformerTokenizer, tokenized_items, args):
    """
    Main function that chooses between single and multi-GPU encoding
    """
    if torch.cuda.device_count() > 1 and args.use_multi_gpu:
        return encode_all_items_multi_gpu(model, tokenizer, tokenized_items, args)
    else:
        return encode_all_items_single_gpu(model, tokenizer, tokenized_items, args)


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

def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, args):

    model.train()

    for step, batch in enumerate(tqdm(dataloader, ncols=100, desc='Training')):
        for k, v in batch.items():
            batch[k] = v.to(args.device)

        if args.fp16:
            with autocast():
                loss = model(**batch)
        else:
            loss = model(**batch)

        # Handle DataParallel loss (already averaged across GPUs)
        if isinstance(model, nn.DataParallel):
            # DataParallel automatically averages the loss across GPUs
            loss = loss.mean() if loss.dim() > 0 else loss

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.fp16:
            print("LOSSSS", loss)
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:

                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                optimizer_was_run = scale_before <= scale_after
                optimizer.zero_grad()

                if optimizer_was_run:
                    scheduler.step()

            else:

                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                optimizer.zero_grad()

tokenizer_glb = None  # must be global for child processes

def init_tokenizer(model_name_or_path, finetune_negative_sample_size, item2id_len):
    global tokenizer_glb
    config = RecformerConfig.from_pretrained(model_name_or_path)
    config.max_attr_num = 3
    config.max_attr_length = 32
    config.max_item_embeddings = 51
    config.attention_window = [64] * 12
    config.max_token_num = 1024
    config.item_num = item2id_len
    config.finetune_negative_sample_size = finetune_negative_sample_size
    tokenizer_glb = RecformerTokenizer.from_pretrained(model_name_or_path, config)

def _par_tokenize_doc(doc):
    global tokenizer_glb
    item_id, item_attr = doc
    input_ids, token_type_ids = tokenizer_glb.encode_item(item_attr)
    return item_id, input_ids, token_type_ids

def main():
    parser = ArgumentParser()
    # path and file
    parser.add_argument('--pretrain_ckpt', type=str, default= "longformer_ckpt/recformer_seqrec_ckpt.bin")
    parser.add_argument('--data_path', type=str, default="finetune_data")
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--ckpt', type=str, default='best_model.bin')
    parser.add_argument('--model_name_or_path', type=str, default='allenai/longformer-base-4096')
    parser.add_argument('--train_file', type=str, default='train.json')
    parser.add_argument('--dev_file', type=str, default='val.json')
    parser.add_argument('--test_file', type=str, default='test.json')
    parser.add_argument('--item2id_file', type=str, default='smap.json')
    parser.add_argument('--meta_file', type=str, default='meta_data.json')

    # data process
    parser.add_argument('--preprocessing_num_workers', type=int, default=8, help="The number of processes to use for the preprocessing.")
    parser.add_argument('--dataloader_num_workers', type=int, default=0)

    # model
    parser.add_argument('--temp', type=float, default=0.05, help="Temperature for softmax.")

    # train
    parser.add_argument('--num_train_epochs', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--finetune_negative_sample_size', type=int, default=1000)
    parser.add_argument('--metric_ks', nargs='+', type=int, default=[10, 50], help='ks for Metric@k')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fix_word_embedding', action='store_true')
    parser.add_argument('--verbose', type=int, default=3)
    
    # Multi-GPU options
    parser.add_argument('--use_multi_gpu', action='store_true', help='Use multiple GPUs for item encoding')
    

    args = parser.parse_args()
    
    print(args)
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    seed_everything(42)
    args.device = torch.device('cuda:0') if args.device>=0 else torch.device('cpu')

    train, val, test, item_meta_dict, item2id, id2item = load_data(args)

    config = RecformerConfig.from_pretrained(args.model_name_or_path)
    config.max_attr_num = 3
    config.max_attr_length = 32
    config.max_item_embeddings = 51
    config.attention_window = [64] * 12
    config.max_token_num = 1024
    config.item_num = len(item2id)
    config.finetune_negative_sample_size = args.finetune_negative_sample_size
    tokenizer = RecformerTokenizer.from_pretrained(args.model_name_or_path, config)
    
    global tokenizer_glb
    tokenizer_glb = tokenizer

    path_corpus = Path(args.data_path)
    dir_preprocess = path_corpus / 'preprocess'
    dir_preprocess.mkdir(exist_ok=True)

    path_output = Path(args.output_dir) / path_corpus.name
    path_output.mkdir(exist_ok=True, parents=True)
    path_ckpt = path_output / args.ckpt

    path_tokenized_items = dir_preprocess / f'tokenized_items_{path_corpus.name}'

    if path_tokenized_items.exists():
        print(f'[Preprocessor] Use cache: {path_tokenized_items}')
    else:
        print(f'Loading attribute data {path_corpus}')

        pool = Pool(
            processes=args.preprocessing_num_workers,
            initializer=init_tokenizer,
            initargs=(args.model_name_or_path, args.finetune_negative_sample_size, len(item2id))
        )
        pool_func = pool.imap(func=_par_tokenize_doc, iterable=item_meta_dict.items())

    
        doc_tuples = list(tqdm(pool_func, total=len(item_meta_dict), ncols=100, desc=f'[Tokenize] {path_corpus}'))
        tokenized_items = {item2id[item_id]: [input_ids, token_type_ids] for item_id, input_ids, token_type_ids in doc_tuples}
        pool.close()
        pool.join()

        torch.save(tokenized_items, path_tokenized_items)

    tokenized_items = torch.load(path_tokenized_items)
    print(f'Successfully load {len(tokenized_items)} tokenized items.')

    finetune_data_collator = FinetuneDataCollatorWithPadding(tokenizer, tokenized_items)
    eval_data_collator = EvalDataCollatorWithPadding(tokenizer, tokenized_items)

    train_data = RecformerTrainDataset(train, collator=finetune_data_collator)
    val_data = RecformerEvalDataset(train, val, test, mode='val', collator=eval_data_collator)
    test_data = RecformerEvalDataset(train, val, test, mode='test', collator=eval_data_collator)

    
    train_loader = DataLoader(train_data, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              collate_fn=train_data.collate_fn)
    dev_loader = DataLoader(val_data, 
                            batch_size=args.batch_size, 
                            collate_fn=val_data.collate_fn)
    test_loader = DataLoader(test_data, 
                            batch_size=args.batch_size, 
                            collate_fn=test_data.collate_fn)

    model = RecformerForSeqRec(config)
    pretrain_ckpt = torch.load(args.pretrain_ckpt)
    model.load_state_dict(pretrain_ckpt, strict=False)
    model.to(args.device)
    
    # Move model to primary device first
    if args.use_multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model)  # Then wrap with DataParallel

    if args.fix_word_embedding:
        print('Fix word embeddings.')
        base_model = get_base_model(model)
        for param in base_model.longformer.embeddings.word_embeddings.parameters():
            param.requires_grad = False

    path_item_embeddings = dir_preprocess / f'item_embeddings_{path_corpus.name}'
    if path_item_embeddings.exists():
        print(f'[Item Embeddings] Use cache: {path_item_embeddings}')
    else:
        print(f'Encoding items.')
        base_model = get_base_model(model)
        item_embeddings = encode_all_items(base_model.longformer, tokenizer, tokenized_items, args)
        torch.save(item_embeddings, path_item_embeddings)
            
    item_embeddings = torch.load(path_item_embeddings)
       
    # Initialize item embeddings properly for DataParallel
    base_model = get_base_model(model)
    base_model.init_item_embedding(item_embeddings)

    model.to(args.device) # send item embeddings to device

    num_train_optimization_steps = int(len(train_loader) / args.gradient_accumulation_steps) * args.num_train_epochs
    optimizer, scheduler = create_optimizer_and_scheduler(model, num_train_optimization_steps, args)
    
    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    test_metrics = eval(model, test_loader, args)
    print(f'Test set: {test_metrics}')
    
    best_target = float('-inf')
    patient = 5

    for epoch in range(args.num_train_epochs):

        base_model = get_base_model(model)
        item_embeddings = encode_all_items(base_model.longformer, tokenizer, tokenized_items, args)
        base_model.init_item_embedding(item_embeddings)

        train_one_epoch(model, train_loader, optimizer, scheduler, scaler, args)
        
        if (epoch + 1) % args.verbose == 0:
            dev_metrics = eval(model, dev_loader, args)
            print(f'Epoch: {epoch}. Dev set: {dev_metrics}')

            if dev_metrics['NDCG@10'] > best_target:
                print('Save the best model.')
                best_target = dev_metrics['NDCG@10']
                patient = 5
                
                base_model = get_base_model(model)
                torch.save(base_model.state_dict(), path_ckpt)
            
            else:
                patient -= 1
                if patient == 0:
                    break

    print('Load best model in stage 1.')
    checkpoint = torch.load(path_ckpt)
    base_model = get_base_model(model)
    base_model.load_state_dict(checkpoint)


    patient = 3

    for epoch in range(args.num_train_epochs):

        train_one_epoch(model, train_loader, optimizer, scheduler, scaler, args)
        
        if (epoch + 1) % args.verbose == 0:
            dev_metrics = eval(model, dev_loader, args)
            print(f'Epoch: {epoch}. Dev set: {dev_metrics}')

            if dev_metrics['NDCG@10'] > best_target:
                print('Save the best model.')
                best_target = dev_metrics['NDCG@10']
                patient = 3
                base_model = get_base_model(model)
                torch.save(base_model.state_dict(), path_ckpt)
            
            else:
                patient -= 1
                if patient == 0:
                    break


    # Final test
    print('Test with the best checkpoint.')  
    checkpoint = torch.load(path_ckpt)
    base_model = get_base_model(model)
    base_model.load_state_dict(checkpoint)
    test_metrics = eval(model, test_loader, args)
    print(f'Test set: {test_metrics}')
               
if __name__ == "__main__":
    main()