import os
import json
import torch
from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score
from pytorch_lightning import seed_everything

from utils import read_json
from optimization import create_optimizer_and_scheduler
from recformer import RecformerModel, RecformerTokenizer, RecformerConfig, RecformerForFraudDetection
from collator import EvalDataCollatorWithPadding
from dataloader import RecformerFraudDataset


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


def encode_all_items(model: RecformerModel, tokenizer: RecformerTokenizer, tokenized_items, args):

    model.eval()

    items = sorted(list(tokenized_items.items()), key=lambda x: x[0])
    items = [ele[1] for ele in items]

    item_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(items), args.batch_size), ncols=100, desc='Encode all items'):

            item_batch = [[item] for item in items[i:i+args.batch_size]]

            inputs = tokenizer.batch_encode(item_batch, encode_item=False)

            for k, v in inputs.items():
                inputs[k] = torch.LongTensor(v).to(args.device)

            outputs = model(**inputs)

            item_embeddings.append(outputs.pooler_output.detach())

    item_embeddings = torch.cat(item_embeddings, dim=0)#.cpu()

    return item_embeddings


def eval_fraud(model, dataloader, args):
    """Evaluate fraud detection model"""
    
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, ncols=100, desc='Evaluate'):
        
        # Move batch to device
        input_ids = batch['input_ids'].to(args.device)
        attention_mask = batch['attention_mask'].to(args.device)
        labels = batch['labels'].to(args.device)
        
        if 'global_attention_mask' in batch:
            global_attention_mask = batch['global_attention_mask'].to(args.device)
        else:
            global_attention_mask = None

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                labels=labels,
                return_dict=True
            )
            
            loss = outputs['loss']
            logits = outputs['logits']
            
            # Get predictions and probabilities
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Fraud probability
            
            total_loss += loss.item()
            num_batches += 1

    # Calculate metrics
    avg_loss = total_loss / num_batches
    accuracy = accuracy_score(all_labels, all_predictions)
    balanced_accuracy = balanced_accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    auc = roc_auc_score(all_labels, all_probabilities)

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        "balanced_accuracy": balanced_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

    return metrics

def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, args):
    """Train one epoch for fraud detection"""
    model.train()
    
    total_loss = 0.0
    num_batches = 0

    for step, batch in enumerate(tqdm(dataloader, ncols=100, desc='Training')):
        for k, v in batch.items():
            batch[k] = v.to(args.device)

        if args.fp16:
            with autocast():
                outputs = model(**batch)
        else:
            outputs = model(**batch)

        loss = outputs['loss']

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss += loss.item()
        num_batches += 1

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
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()

    avg_loss = total_loss / num_batches
    return avg_loss

tokenizer_glb = None  # must be global for child processes

def init_tokenizer(model_name_or_path, item2id_len):
    global tokenizer_glb
    config = RecformerConfig.from_pretrained(model_name_or_path)
    config.max_attr_num = 3
    config.max_attr_length = 32
    config.max_item_embeddings = 51
    config.attention_window = [64] * 12
    config.max_token_num = 1024
    config.item_num = item2id_len
    tokenizer_glb = RecformerTokenizer.from_pretrained(model_name_or_path, config)

def _par_tokenize_doc(doc):
    global tokenizer_glb
    item_id, item_attr = doc
    input_ids, token_type_ids = tokenizer_glb.encode_item(item_attr)
    return item_id, input_ids, token_type_ids


def calculate_pos_weight_from_dataset(dataset: DataLoader, label_key:str="label"):
    """
    Calculates fraud ratio, non-fraud ratio, and pos_weight for binary classification.
    
    Args:
        dataset: torch.utils.data.Dataset (assumed to return dicts or tuples containing labels)
        label_key: key for accessing labels if dataset returns dicts (default: "label")

    Returns:
        fraud_ratio (float), non_fraud_ratio (float), pos_weight (float)
    """
    labels = []

    for batch in dataset:
        batch_labels = batch[label_key] if isinstance(batch, dict) else batch[1]
        labels.extend(batch_labels.tolist())

    labels = torch.tensor(labels, dtype=torch.float32)
    fraud_ratio = labels.mean().item()
    non_fraud_ratio = 1.0 - fraud_ratio
    pos_weight = non_fraud_ratio / fraud_ratio if fraud_ratio > 0 else 0.0

    print(
        f"Class distribution - Non-fraud: {non_fraud_ratio:.5f}, Fraud: {fraud_ratio:.5f}"
    )
    print(f"Using pos_weight: {pos_weight:.2f} for fraud class")

    return pos_weight
 
def main():
    parser = ArgumentParser()
    # path and file
    parser.add_argument('--pretrain_ckpt', type=str, default= "pretrain_ckpt/fraud_pretrain_ckpt.bin")
    parser.add_argument('--data_path', type=str, default="transactional_data_process/classification_data")
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
    parser.add_argument('--metric_ks', nargs='+', type=int, default=[10, 50], help='ks for Metric@k')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fix_word_embedding', action='store_true')
    parser.add_argument('--verbose', type=int, default=3)
    

    args = parser.parse_args()
    print(args)
    seed_everything(42)
    args.device = torch.device('cuda:{}'.format(args.device)) if args.device>=0 else torch.device('cpu')

    train, val, test, item_meta_dict, item2id, id2item = load_data(args)

    config = RecformerConfig.from_pretrained(args.model_name_or_path)
    config.max_attr_num = 3
    config.max_attr_length = 32
    config.max_item_embeddings = 51
    config.attention_window = [64] * 12
    config.max_token_num = 1024
    config.item_num = len(item2id)
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
            initargs=(args.model_name_or_path, len(item2id))
        )
        pool_func = pool.imap(func=_par_tokenize_doc, iterable=item_meta_dict.items())

    
        doc_tuples = list(tqdm(pool_func, total=len(item_meta_dict), ncols=100, desc=f'[Tokenize] {path_corpus}'))
        tokenized_items = {item2id[item_id]: [input_ids, token_type_ids] for item_id, input_ids, token_type_ids in doc_tuples}
        pool.close()
        pool.join()

        torch.save(tokenized_items, path_tokenized_items)

    tokenized_items = torch.load(path_tokenized_items)
    print(f'Successfully load {len(tokenized_items)} tokenized items.')

    #prepare data 
    finetune_data_collator = EvalDataCollatorWithPadding(tokenizer, tokenized_items)

    train_data = RecformerFraudDataset(train, collator=finetune_data_collator)
    val_data = RecformerFraudDataset(val, collator=finetune_data_collator)
    test_data = RecformerFraudDataset(test, collator=finetune_data_collator)
    
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
    
    #account for the imbalance in the loss
    config.pos_weight = calculate_pos_weight_from_dataset(train_loader)

    #load the model
    model = RecformerForFraudDetection(config)
    pretrain_ckpt = torch.load(args.pretrain_ckpt)
    model.load_state_dict(pretrain_ckpt, strict=False)
    model.to(args.device)
    
    #encode all the items (tokenise the transaction meta data)
    path_item_embeddings = dir_preprocess / f'item_embeddings_{path_corpus.name}'
    if path_item_embeddings.exists():
        print(f'[Item Embeddings] Use cache: {path_tokenized_items}')
    else:
        print(f'Encoding items.')
        item_embeddings = encode_all_items(model.longformer, tokenizer, tokenized_items, args)
        torch.save(item_embeddings, path_item_embeddings)
    
    item_embeddings = torch.load(path_item_embeddings)
    model.init_item_embedding(item_embeddings)
    model.to(args.device) # send item embeddings to device

    num_train_optimization_steps = int(len(train_loader) / args.gradient_accumulation_steps) * args.num_train_epochs
    optimizer, scheduler = create_optimizer_and_scheduler(model, num_train_optimization_steps, args)
    
    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    test_metrics = eval_fraud(model, test_loader, args)
    print(f'Test set: {test_metrics}')
    
    best_target = float('-inf')
    patience_counter = 0
    patience = 5
    epoch_metrics_log = []
    
    #Start training
    for epoch in range(args.num_train_epochs):

        #encode the items for each epoch
        item_embeddings = encode_all_items(model.longformer, tokenizer, tokenized_items, args)
        model.init_item_embedding(item_embeddings)

        # Train an epoch
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, args)
        print(f'Average training loss: {avg_train_loss:.4f}')

        if (epoch + 1) % args.verbose == 0:
            dev_metrics = eval_fraud(model, dev_loader, args)
            print(f'Epoch: {epoch}. Dev set: {dev_metrics}')
            # Save metrics for this epoch
            log_entry = {
                "epoch": epoch + 1,
                "avg_train_loss": avg_train_loss,
                **dev_metrics
            }
            epoch_metrics_log.append(log_entry)

            #save best model on balanced accuracy          
            if dev_metrics['balanced_accuracy'] > best_target:
                print('Save the best model.')
                best_target = dev_metrics['balanced_accuracy']
                patience_counter = 0
                torch.save(model.state_dict(), path_ckpt)
            
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping after {patience} epochs without improvement')
                    break
    
    metrics_path = path_output / "epoch_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(epoch_metrics_log, f, indent=4)
    print(f"Saved epoch metrics to {metrics_path}")

    print('Test with the best checkpoint.')  
    model.load_state_dict(torch.load(path_ckpt))
    test_metrics = eval_fraud(model, test_loader, args)
    print(f'Test set: {test_metrics}')
    
    test_path = path_output / "test_metrics.json"
    with open(test_path, 'w') as f:
        json.dump(test_metrics, f, indent=4)
    print(f"Saved test metrics to {test_path}")

               
if __name__ == "__main__":
    main()