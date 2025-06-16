import torch
from collections import OrderedDict
from recformer import RecformerModel, RecformerConfig, RecformerForSeqRec, RecformerForFraudDetection


import torch
from collections import OrderedDict
from recformer import RecformerModel, RecformerConfig, RecformerForSeqRec, RecformerForFraudDetection

PRETRAINED_CKPT_PATH = 'pretrain_ckpt/pytorch_model.bin'
LONGFORMER_CKPT_PATH = 'longformer_ckpt/longformer-base-4096.bin'
LONGFORMER_TYPE = 'allenai/longformer-base-4096'
RECFORMER_OUTPUT_PATH = 'pretrain_ckpt/recformer_pretrain_ckpt.bin'
RECFORMERSEQREC_OUTPUT_PATH = 'pretrain_ckpt/seqrec_pretrain_ckpt.bin'
RECFORMERFRAUD_OUTPUT_PATH = 'pretrain_ckpt/fraud_pretrain_ckpt.bin'

input_file = PRETRAINED_CKPT_PATH
state_dict = torch.load(input_file)

print("\nCheckpoint keys preview:")
for key in list(state_dict.keys())[:20]:  # show just the first 20
    print(key)

# Get the vocabulary size from your pretrained model
pretrained_vocab_size = state_dict['_forward_module.model.longformer.embeddings.word_embeddings.weight'].shape[0]
print(f"Pretrained model vocabulary size: {pretrained_vocab_size}")

# Load longformer state dict for reference (but don't overwrite embeddings)
longformer_file = LONGFORMER_CKPT_PATH
longformer_state_dict = torch.load(longformer_file)
longformer_vocab_size = longformer_state_dict['longformer.embeddings.word_embeddings.weight'].shape[0]
print(f"Original Longformer vocabulary size: {longformer_vocab_size}")

# DON'T overwrite the word embeddings - your pretrained model already has the correct vocab size
# REMOVE this line: state_dict['_forward_module.model.longformer.embeddings.word_embeddings.weight'] = longformer_state_dict['longformer.embeddings.word_embeddings.weight']

# === Convert to RecformerModel ===
output_file = RECFORMER_OUTPUT_PATH
new_state_dict = OrderedDict()
for key, value in state_dict.items():
    if key.startswith('_forward_module.model.longformer.'):
        new_key = key[len('_forward_module.model.longformer.'):]
        new_state_dict[new_key] = value

# Create config with correct vocabulary size
config = RecformerConfig.from_pretrained(LONGFORMER_TYPE)
config.vocab_size = pretrained_vocab_size  # Use the vocabulary size from your pretrained model
config.max_attr_num = 3
config.max_attr_length = 32
config.max_item_embeddings = 51
config.attention_window = [64] * 12

model = RecformerModel(config)
model.resize_token_embeddings(config.vocab_size)  # Resize to match config
missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
print(f'RecformerModel convert successfully. Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}')
torch.save(new_state_dict, output_file)

# === Convert to RecformerForSeqRec ===
output_file = RECFORMERSEQREC_OUTPUT_PATH
new_state_dict = OrderedDict()
for key, value in state_dict.items():
    if key.startswith('_forward_module.model.'):
        new_key = key[len('_forward_module.model.'):]
        new_state_dict[new_key] = value

# Create config with correct vocabulary size
config = RecformerConfig.from_pretrained(LONGFORMER_TYPE)
config.vocab_size = pretrained_vocab_size  # Use the vocabulary size from your pretrained model
config.max_attr_num = 3
config.max_attr_length = 32
config.max_item_embeddings = 51
config.attention_window = [64] * 12

model = RecformerForSeqRec(config)
model.resize_token_embeddings(config.vocab_size)  # Resize to match config
missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
print(f'RecformerForSeqRec convert successfully. Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}')
torch.save(new_state_dict, output_file)

# === Convert to RecformerForFraudDetection ===
output_file = RECFORMERFRAUD_OUTPUT_PATH
new_state_dict = OrderedDict()
for key, value in state_dict.items():
    if key.startswith('_forward_module.model.'):
        new_key = key[len('_forward_module.model.'):]
        new_state_dict[new_key] = value

# Create config with correct vocabulary size
config = RecformerConfig.from_pretrained(LONGFORMER_TYPE)
config.vocab_size = pretrained_vocab_size  # Use the vocabulary size from your pretrained model
config.max_attr_num = 3
config.max_attr_length = 32
config.max_item_embeddings = 51
config.attention_window = [64] * 12

model = RecformerForFraudDetection(config)
model.resize_token_embeddings(config.vocab_size)  # Resize to match config
missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
print(f'RecformerForFraudDetection convert successfully. Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}')
torch.save(new_state_dict, output_file)

# PRETRAINED_CKPT_PATH = 'pretrain_ckpt/pytorch_model.bin'
# LONGFORMER_CKPT_PATH = 'longformer_ckpt/longformer-base-4096.bin'
# LONGFORMER_TYPE = 'allenai/longformer-base-4096'
# RECFORMER_OUTPUT_PATH = 'pretrain_ckpt/recformer_pretrain_ckpt.bin'
# RECFORMERSEQREC_OUTPUT_PATH = 'pretrain_ckpt/seqrec_pretrain_ckpt.bin'
# RECFORMERFRAUD_OUTPUT_PATH = 'pretrain_ckpt/fraud_pretrain_ckpt.bin'


# input_file = PRETRAINED_CKPT_PATH
# state_dict = torch.load(input_file)

# longformer_file = LONGFORMER_CKPT_PATH
# longformer_state_dict = torch.load(longformer_file)

# state_dict['_forward_module.model.longformer.embeddings.word_embeddings.weight'] = longformer_state_dict['longformer.embeddings.word_embeddings.weight']

# # === Convert to RecformerModel ===
# output_file = RECFORMER_OUTPUT_PATH
# new_state_dict = OrderedDict()

# for key, value in state_dict.items():

#     if key.startswith('_forward_module.model.longformer.'):
#         new_key = key[len('_forward_module.model.longformer.'):]
#         new_state_dict[new_key] = value

# config = RecformerConfig.from_pretrained(LONGFORMER_TYPE)
# config.max_attr_num = 3
# config.max_attr_length = 32
# config.max_item_embeddings = 51
# config.attention_window = [64] * 12
# model = RecformerModel(config)
# model.load_state_dict(new_state_dict)

# print('RecformerModel convert successfully.')
# torch.save(new_state_dict, output_file)


# # === Convert to RecformerForSeqRec ===
# output_file = RECFORMERSEQREC_OUTPUT_PATH
# new_state_dict = OrderedDict()

# for key, value in state_dict.items():

#     if key.startswith('_forward_module.model.'):
#         new_key = key[len('_forward_module.model.'):]
#         new_state_dict[new_key] = value

# config = RecformerConfig.from_pretrained(LONGFORMER_TYPE)
# config.max_attr_num = 3
# config.max_attr_length = 32
# config.max_item_embeddings = 51
# config.attention_window = [64] * 12
# model = RecformerForSeqRec(config)

# model.load_state_dict(new_state_dict, strict=False)

# print('RecformerForSeqRec convert successfully.')
# torch.save(new_state_dict, output_file)


# # === Convert to RecformerForFraudDetection ===
# output_file = RECFORMERFRAUD_OUTPUT_PATH
# new_state_dict = OrderedDict()

# for key, value in state_dict.items():
#     if key.startswith('_forward_module.model.'):
#         new_key = key[len('_forward_module.model.'):]
#         new_state_dict[new_key] = value

# config = RecformerConfig.from_pretrained(LONGFORMER_TYPE)
# config.max_attr_num = 3
# config.max_attr_length = 32
# config.max_item_embeddings = 51
# config.attention_window = [64] * 12

# model = RecformerForFraudDetection(config)
# model.load_state_dict(new_state_dict, strict=False)

# print('RecformerForFraudDetection convert successfully.')
# torch.save(new_state_dict, output_file)



