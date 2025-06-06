import gzip
import json
from tqdm import tqdm
import os

META_ROOT = '../data/01_raw' # Set your meta data path
SEQ_ROOT = '../data/01_raw' # Set your seq data path

# pretrain_categories = ['Automotive', 'Cell_Phones_and_Accessories', 'Electronics', 'CDs_and_Vinyl']
pretrain_categories = ['Automotive', 'CDs_and_Vinyl']

pretrain_meta_pathes = [f'{META_ROOT}/{cate}_metadata.jsonl.gz' for cate in pretrain_categories]
pretrain_seq_pathes = [f'{SEQ_ROOT}/{cate}_reviews.jsonl.gz' for cate in pretrain_categories]

for path in pretrain_meta_pathes+pretrain_seq_pathes:
    if not os.path.exists(path):
        print(path)
        exit(0)

def extract_meta_data(path, meta_data, selected_asins):
    title_length = 0
    total_num = 0
    with gzip.open(path) as f:
        print()
        for line in tqdm(f, ncols=100):
            line = json.loads(line)
            attr_dict = dict()
            asin = line['asin']
            if asin not in selected_asins:
                continue
            
            category = ' '.join(line['category'])
            brand = line['brand']
            title = line['title']

            title_length += len(title.split())
            total_num += 1

            attr_dict['title'] = title
            attr_dict['brand'] = brand
            attr_dict['category'] = category
            meta_data[asin] = attr_dict   
    return title_length, total_num    


meta_asins = set()
seq_asins = set()

for path in tqdm(pretrain_meta_pathes, ncols=100, desc='Check meta asins'):
    with gzip.open(path) as f:
        for line in f:
            line = json.loads(line)
            if 'asin' not in line or 'title' not in line:
                print(f'Error in {path}: {line.keys()}')
                exit(0)
            if line['asin'] is not None and line['title'] is not None:
                meta_asins.add(line['asin'])

for path in tqdm(pretrain_seq_pathes, ncols=100, desc='Check seq asins'):
    with gzip.open(path) as f:
        for line in f:
            line = json.loads(line)
            if line['asin'] is not None and line['reviewerID'] is not None:
                seq_asins.add(line['asin'])


# MAX_ASIN_NUM = 13369  # Set a limit for the number of asins to select
selected_asins = meta_asins & seq_asins
# selected_asins = list(selected_asins)[:MAX_ASIN_NUM]  # Limit the number of selected asins

print(f'Meta has {len(meta_asins)} Asins.')
print(f'Seq has {len(seq_asins)} Asins.')
print(f'{len(selected_asins)} Asins are selected.')

meta_data = dict()
for path in tqdm(pretrain_meta_pathes, ncols=100, desc=path):
    t_l, t_n = extract_meta_data(path, meta_data, selected_asins)
    print(f'Average title length of {path}', t_l/t_n)

with open('meta_data.json', 'w', encoding='utf8') as f:
    json.dump(meta_data, f)