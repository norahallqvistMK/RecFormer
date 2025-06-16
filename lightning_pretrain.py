from torch.utils.tensorboard import SummaryWriter
import logging
from torch.utils.data import DataLoader
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
from multiprocessing import get_context
import json
import argparse
import torch
from functools import partial
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from functools import partial

from recformer import RecformerForPretraining, RecformerTokenizer, RecformerConfig, LitWrapper
from collator import PretrainDataCollatorWithPadding
from lightning_dataloader import ClickDataset

logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, default=None)
parser.add_argument('--temp', type=float, default=0.05, help="Temperature for softmax.")
parser.add_argument('--preprocessing_num_workers', type=int, default=8, help="The number of processes to use for the preprocessing.")
parser.add_argument('--train_file', type=str, required=True)
parser.add_argument('--dev_file', type=str, required=True)
parser.add_argument('--item_attr_file', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--num_train_epochs', type=int, default=10)
parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
parser.add_argument('--dataloader_num_workers', type=int, default=2)
parser.add_argument('--mlm_probability', type=float, default=0.15)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=5e-5)
parser.add_argument('--valid_step', type=int, default=2000)
parser.add_argument('--log_step', type=int, default=2000)
parser.add_argument('--device', type=int, default=1)
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--longformer_ckpt', type=str, default='longformer_ckpt/longformer-base-4096.bin')
parser.add_argument('--fix_word_embedding', action='store_true')

def _par_tokenize_doc(doc, tokenizer):
    item_id, item_attr = doc
    # print(f'Tokenizing item {item_id} with attributes {item_attr}')
    input_ids, token_type_ids = tokenizer.encode_item(item_attr)

    return item_id, input_ids, token_type_ids

def main():
    
    args = parser.parse_args()
    print(args)
    seed_everything(42)

    config = RecformerConfig.from_pretrained(args.model_name_or_path)
    config.max_attr_num = 3
    config.max_attr_length = 32
    config.max_item_embeddings = 51  # 50 item and 1 for cls
    config.attention_window = [64] * 12
    config.max_token_num = 1024
    tokenizer = RecformerTokenizer.from_pretrained(args.model_name_or_path, config)

    # global tokenizer_glb
    print(f'Using tokenizer tokenizer_glb: {tokenizer}')
    print(f'Tokenizer vocabulary size: {len(tokenizer)}')

    # preprocess corpus
    path_corpus = Path(args.item_attr_file)
    dir_corpus = path_corpus.parent
    dir_preprocess = dir_corpus / 'preprocess'
    dir_preprocess.mkdir(exist_ok=True)

    path_tokenized_items = dir_preprocess / f'tokenized_items_{path_corpus.name}'

    if path_tokenized_items.exists():
        print(f'[Preprocessor] Use cache: {path_tokenized_items}')
    else:
        print(f'Loading attribute data {path_corpus}')
        item_attrs = json.load(open(path_corpus))
        pool = Pool(processes=args.preprocessing_num_workers)
        # tokenize_func = partial(_par_tokenize_doc, model_name_or_path=args.model_name_or_path, config = config)
        # pool_func = pool.imap(func=tokenize_func, iterable=item_attrs.items())
        # doc_tuples = list(tqdm(pool_func, total=len(item_attrs), ncols=100, desc=f'[Tokenize] {path_corpus}'))
        tokenize_func = partial(_par_tokenize_doc, tokenizer=tokenizer)
        doc_tuples = []
        for doc in tqdm(item_attrs.items(), total=len(item_attrs), ncols=100, desc=f'[Tokenize] {path_corpus}'):
            result = tokenize_func(doc)
            doc_tuples.append(result)

        tokenized_items = {item_id: [input_ids, token_type_ids] for item_id, input_ids, token_type_ids in doc_tuples}
        pool.close()
        pool.join()

        json.dump(tokenized_items, open(path_tokenized_items, 'w'))

    tokenized_items = json.load(open(path_tokenized_items))#dir_preprocess / f'attr_small.json'))#
    print(f'Successfully load {len(tokenized_items)} tokenized items.')

    tokenized_items = {str(k): v for k, v in tokenized_items.items()}

    data_collator = PretrainDataCollatorWithPadding(tokenizer, tokenized_items, mlm_probability=args.mlm_probability)
    train_data = ClickDataset(json.load(open(args.train_file)), data_collator)
    dev_data = ClickDataset(json.load(open(args.dev_file)), data_collator)

    train_loader = DataLoader(train_data, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              collate_fn=train_data.collate_fn,
                              num_workers=args.dataloader_num_workers)
    dev_loader = DataLoader(dev_data, 
                            batch_size=args.batch_size, 
                            collate_fn=dev_data.collate_fn,
                            num_workers=args.dataloader_num_workers)
        
    # pytorch_model = RecformerForPretraining(config)
    # print(f"Model created with vocab size: {pytorch_model.longformer.embeddings.word_embeddings.num_embeddings}")
    
    # # Load the checkpoint first
    # checkpoint = torch.load(args.longformer_ckpt)
    # old_vocab_size = checkpoint['longformer.embeddings.word_embeddings.weight'].shape[0]
    # new_vocab_size = config.vocab_size
    
    # print(f"Old vocab size: {old_vocab_size}, New vocab size: {new_vocab_size}")
    
    # if old_vocab_size != new_vocab_size:
    #     print("Resizing vocabulary-dependent parameters...")
        
    #     # Resize word embeddings
    #     old_word_embeddings = checkpoint['longformer.embeddings.word_embeddings.weight']
    #     new_word_embeddings = torch.zeros(new_vocab_size, old_word_embeddings.shape[1])
    #     new_word_embeddings[:old_vocab_size] = old_word_embeddings
        
    #     # Initialize new embeddings with small random values based on existing embeddings
    #     if new_vocab_size > old_vocab_size:
    #         mean = old_word_embeddings.mean(dim=0)
    #         std = old_word_embeddings.std(dim=0)
    #         num_new_tokens = new_vocab_size - old_vocab_size
    #         new_embeddings = torch.normal(
    #             mean.unsqueeze(0).expand(num_new_tokens, -1), 
    #             std.unsqueeze(0).expand(num_new_tokens, -1) * 0.02
    #         )
    #         new_word_embeddings[old_vocab_size:] = new_embeddings
        
    #     checkpoint['longformer.embeddings.word_embeddings.weight'] = new_word_embeddings
        
    #     # Resize LM head parameters if they exist
    #     if 'lm_head.decoder.weight' in checkpoint:
    #         old_lm_weight = checkpoint['lm_head.decoder.weight']
    #         new_lm_weight = torch.zeros(new_vocab_size, old_lm_weight.shape[1])
    #         new_lm_weight[:old_vocab_size] = old_lm_weight
            
    #         # Initialize new LM head weights (usually tied to embeddings)
    #         if new_vocab_size > old_vocab_size:
    #             new_lm_weight[old_vocab_size:] = new_embeddings
            
    #         checkpoint['lm_head.decoder.weight'] = new_lm_weight
        
    #     if 'lm_head.bias' in checkpoint:
    #         old_lm_bias = checkpoint['lm_head.bias']
    #         new_lm_bias = torch.zeros(new_vocab_size)
    #         new_lm_bias[:old_vocab_size] = old_lm_bias
    #         # New bias tokens initialized to 0 (default)
    #         checkpoint['lm_head.bias'] = new_lm_bias
        
    #     if 'lm_head.decoder.bias' in checkpoint:
    #         old_decoder_bias = checkpoint['lm_head.decoder.bias']
    #         new_decoder_bias = torch.zeros(new_vocab_size)
    #         new_decoder_bias[:old_vocab_size] = old_decoder_bias
    #         # New decoder bias tokens initialized to 0 (default)
    #         checkpoint['lm_head.decoder.bias'] = new_decoder_bias
    
    # # Now resize the model's embeddings to match
    # pytorch_model.resize_token_embeddings(config.vocab_size)
    # print(f"Embeddings resized to: {pytorch_model.longformer.embeddings.word_embeddings.num_embeddings}")
    
    # # Load the modified checkpoint
    # missing_keys, unexpected_keys = pytorch_model.load_state_dict(checkpoint, strict=False)
    # print(f"Missing keys: {missing_keys}")
    # print(f"Unexpected keys: {unexpected_keys}")
    
    # pytorch_model = RecformerForPretraining(config)
    # pytorch_model.load_state_dict(torch.load(args.longformer_ckpt), strict=False)
    # old_vocab_size = config.vocab_size
    # new_vocab_size = len(tokenizer)

    # print(f"Original vocab size: {old_vocab_size}")
    # print(f"New vocab size: {new_vocab_size}")
    # print(f"Added {new_vocab_size - old_vocab_size} custom tokens")

    # # 5. Resize token embeddings using inherited method
    # print("Resizing token embeddings...")
    # pytorch_model.longformer.resize_token_embeddings(new_vocab_size)
    
    # # 6. Update config vocab size
    # config.vocab_size = new_vocab_size
    # pytorch_model.config.vocab_size = new_vocab_size

    old_vocab_size = config.vocab_size
    new_vocab_size = len(tokenizer)
    print(f"Original vocab size: {old_vocab_size}")
    print(f"New vocab size: {new_vocab_size}")
    print(f"Added {new_vocab_size - old_vocab_size} custom tokens")

    # 1. Create model with original config
    pytorch_model = RecformerForPretraining(config)

    # 2. Load pretrained weights first
    pytorch_model.load_state_dict(torch.load(args.longformer_ckpt), strict=True)

    # 3. Now resize token embeddings (this will preserve old weights and initialize new ones)
    print("Resizing token embeddings...")
    pytorch_model.longformer.resize_token_embeddings(new_vocab_size)

    # 4. The resize_token_embeddings method should handle the LM head automatically
    # But if it doesn't, manually resize it:
    if hasattr(pytorch_model, 'lm_head'):
        old_lm_head_weight = pytorch_model.lm_head.decoder.weight.data
        old_lm_head_bias = pytorch_model.lm_head.decoder.bias.data if pytorch_model.lm_head.decoder.bias is not None else None
        old_lm_bias = pytorch_model.lm_head.bias.data if pytorch_model.lm_head.bias is not None else None
        
        # Create new linear layer
        pytorch_model.lm_head.decoder = torch.nn.Linear(
            pytorch_model.lm_head.decoder.in_features,
            new_vocab_size,
            bias=pytorch_model.lm_head.decoder.bias is not None
        )
        
        if hasattr(pytorch_model.lm_head, 'bias') and pytorch_model.lm_head.bias is not None:
            pytorch_model.lm_head.bias = torch.nn.Parameter(torch.zeros(new_vocab_size))
        
        # Copy old weights
        with torch.no_grad():
            pytorch_model.lm_head.decoder.weight[:old_vocab_size] = old_lm_head_weight
            if old_lm_head_bias is not None:
                pytorch_model.lm_head.decoder.bias[:old_vocab_size] = old_lm_head_bias
            if old_lm_bias is not None:
                pytorch_model.lm_head.bias[:old_vocab_size] = old_lm_bias

    # 5. Update config
    config.vocab_size = new_vocab_size
    pytorch_model.config.vocab_size = new_vocab_size

    print("Model loaded and resized successfully!")

    if args.fix_word_embedding:
        print('Fix word embeddings.')
        for param in pytorch_model.longformer.embeddings.word_embeddings.parameters():
            param.requires_grad = True

    model = LitWrapper(pytorch_model, learning_rate=args.learning_rate)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=5, 
        monitor="accuracy",  # or just "loss" depending on your logging
        mode="max",          # "min" because lower loss is better
        filename="{epoch}-{accuracy:.4f}"
    )     
    for batch in train_loader:
        print(batch)
    
    trainer = Trainer(accelerator="gpu",
                      max_epochs=args.num_train_epochs,
                     devices=args.device,
                     accumulate_grad_batches=args.gradient_accumulation_steps,
                    #  val_check_interval=args.valid_step,
                     default_root_dir=args.output_dir,
                     gradient_clip_val=1.0,
                     log_every_n_steps=args.log_step,
                     precision=16 if args.fp16 else 32,
                     strategy='deepspeed_stage_2',
                     callbacks=[checkpoint_callback]
                     )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=dev_loader, ckpt_path=args.ckpt)

if __name__ == "__main__":
    main()