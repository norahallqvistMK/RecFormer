import torch
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import openai

import os
import json
import torch
from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast


from utils import read_json, AverageMeterSet, Ranker
from optimization import create_optimizer_and_scheduler
from recformer import RecformerModel, RecformerForSeqRec, RecformerTokenizer, RecformerConfig
from collator import FinetuneDataCollatorWithPadding, EvalDataCollatorWithPadding
from dataloader import RecformerTrainDataset, RecformerEvalDataset
import dotenv
dotenv.load_dotenv()


import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def perform_kmeans_clustering(embeddings, k_range, random_state=42):
    """Perform K-means clustering with different k values"""
    clustering_results = {}
    
    for k in tqdm(k_range, desc="K-means clustering"):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Calculate clustering metrics
        inertia = kmeans.inertia_
        silhouette_avg = silhouette_score(embeddings, cluster_labels)
        
        clustering_results[k] = {
            'labels': cluster_labels,
            'centroids': kmeans.cluster_centers_,
            'inertia': inertia,
            'silhouette_score': silhouette_avg,
            'model': kmeans
        }
        
        print(f"K={k}: Inertia={inertia:.2f}, Silhouette Score={silhouette_avg:.4f}")
    
    return clustering_results

def analyze_clustering_results(clustering_results, output_dir=None):
    """Analyze and visualize clustering results"""
    k_values = list(clustering_results.keys())
    inertias = [clustering_results[k]['inertia'] for k in k_values]
    silhouette_scores = [clustering_results[k]['silhouette_score'] for k in k_values]
    
    # Plot elbow curve and silhouette scores
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Elbow curve
    ax1.plot(k_values, inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.grid(True)
    
    # Silhouette scores
    ax2.plot(k_values, silhouette_scores, 'ro-')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score vs Number of Clusters')
    ax2.grid(True)
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'clustering_analysis.png'), dpi=300, bbox_inches='tight')
    else:   
        plt.savefig('clustering_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find optimal k based on silhouette score
    optimal_k = max(clustering_results.keys(), key=lambda k: clustering_results[k]['silhouette_score'])
    print(f"\nOptimal k based on silhouette score: {optimal_k}")
    
    return optimal_k

def visualize_clusters(embeddings, cluster_labels, k, method='tsne', output_dir=None):
    """Visualize clusters in 2D using t-SNE or PCA"""
    
    # Reduce dimensionality for visualization
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)//4))
        embeddings_2d = reducer.fit_transform(embeddings)
    else:  # PCA
        reducer = PCA(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=cluster_labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'K-means Clustering Visualization (k={k}) - {method.upper()}')
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, f'clusters_k{k}_{method}.png'), dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f'clusters_k{k}_{method}.png', dpi=300, bbox_inches='tight')
    
    plt.show()

def save_clustering_results(clustering_results, sequence_ids, output_dir):
    """Save clustering results to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    for k, results in clustering_results.items():
        # Save cluster assignments
        cluster_assignments = {
            'sequence_ids': sequence_ids,
            'cluster_labels': results['labels'].tolist(),
            'silhouette_score': float(results['silhouette_score']),
            'inertia': float(results['inertia'])
        }
        
        with open(os.path.join(output_dir, f'cluster_assignments_k{k}.json'), 'w') as f:
            json.dump(cluster_assignments, f, indent=2)
        
        # Save cluster centroids
        np.save(os.path.join(output_dir, f'centroids_k{k}.npy'), results['centroids'])

def get_cluster_statistics(clustering_results, k):
    """Get detailed statistics for a specific k value"""
    if k not in clustering_results:
        print(f"K={k} not found in clustering results")
        return
    
    labels = clustering_results[k]['labels']
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    print(f"\nCluster statistics for k={k}:")
    print(f"Silhouette Score: {clustering_results[k]['silhouette_score']:.4f}")
    print(f"Inertia: {clustering_results[k]['inertia']:.2f}")
    print("Cluster distribution:")
    
    for label, count in zip(unique_labels, counts):
        percentage = count / len(labels) * 100
        print(f"  Cluster {label}: {count} sequences ({percentage:.2f}%)")

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
    print(tokenizer_glb.config)  # should now print correctly
    input_ids, token_type_ids = tokenizer_glb.encode_item(item_attr)
    return item_id, input_ids, token_type_ids

def get_cluster_description(
    items_in_cluster: List, 
    model: str = "gpt-4",
    temperature: float = 0.7,
    max_tokens: int = 200,
    timeout: int = 30
) -> str:
    """
    Generate a cluster description based on user interaction sequences.
    
    Args:
        items_in_cluster: List of user items in the cluster
        model: OpenAI model to use (default: "gpt-4")
        temperature: Sampling temperature (default: 0.7)
        max_tokens: Maximum tokens in response (default: 200)
        timeout: Request timeout in seconds (default: 30)
        
    Returns:
        str: Generated cluster description
        
    Raises:
        ValueError: If inputs are invalid or environment variables are missing
        Exception: If API call fails
    """
    # Validate inputs
    if not items_in_cluster:
        raise ValueError("Items list cannot be empty")
    
    # Validate environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    # Initialize OpenAI client
    client_kwargs = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url
    
    client = openai.OpenAI(**client_kwargs)

    # Truncate items if too many to avoid token limits
    max_items = 50
    display_items = items_in_cluster[:max_items]
    truncated_note = f"\n\n[Note: Showing first {max_items} of {len(items_in_cluster)} items]" if len(items_in_cluster) > max_items else ""
    
    # Build the analysis prompt
    task = f"""
    You are an expert in analyzing item clusters and generating descriptive summaries.
    
    You are given a list of items from users in the same cluster.
    Each item is described by a set of characteristics, such as item name, category, and other attributes.
    
    Your task is to:
    1. Analyze the common patterns across all items
    2. Identify shared characteristics and themes among the items
    3. Generate a concise cluster description (2-3 sentences) that captures the essence of the grouped items
    4. Focus on what makes this cluster unique and distinguishable from other item groups
    
    Items in Cluster:
    {display_items}{truncated_note}
    
    Please provide only the cluster description without additional explanation.
    """

    try:
        # Make API call with timeout
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert data analyst specializing in user behavior clustering and persona generation."
                },
                {
                    "role": "user", 
                    "content": task
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
            timeout=timeout
        )
        
        # Extract and validate response
        if not response.choices or not response.choices[0].message.content:
            raise Exception("No valid response generated from API")
            
        description = response.choices[0].message.content.strip()
        
        return description
        
    except openai.AuthenticationError as e:
        raise Exception(f"OpenAI authentication failed: {e}")
    except openai.RateLimitError as e:
        raise Exception(f"OpenAI rate limit exceeded: {e}")
    except openai.APITimeoutError as e:
        raise Exception(f"OpenAI API timeout: {e}")
    except openai.APIError as e:
        raise Exception(f"OpenAI API error: {e}")
    except Exception as e:
        if "OpenAI" in str(e):
            raise
        raise Exception(f"Failed to generate cluster description: {e}")

# Modified extract_embeddings function to also collect predictions
def extract_embeddings_with_predictions(model, dataloader, device, max_sequences=None):
    """
    Extract sequence embeddings and predictions from the model using the dataloader.
    
    Args:
        model: The RecformerForSeqRec model
        dataloader: DataLoader containing evaluation data
        device: Device to run inference on
        max_sequences: Maximum number of sequences to process (None for all)
    
    Returns:
        embeddings: numpy array of sequence embeddings
        sequence_ids: list of sequence identifiers
        predictions: list of predicted item IDs for each sequence
    """
    model.eval()
    embeddings = []
    sequence_ids = []
    predictions = []
    
    with torch.no_grad():
        for i, (batch, labels) in enumerate(tqdm(dataloader, desc="Extracting embeddings and predictions")):
            if max_sequences and len(embeddings) >= max_sequences:
                break
                
            # Move batch to device
            for k, v in batch.items():
                batch[k] = v.to(device)

            # Get sequence embeddings (before final classification layer)
            outputs = model.longformer(**batch)
            seq_embeddings = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state[:, 0, :]
            
            # Get predictions by running inference without labels
            logits = model(**batch)  # This returns logits for all items
            predicted_indices = torch.argmax(logits, dim=-1)  # Get the item with highest score
            
            embeddings.append(seq_embeddings.cpu().numpy())
            predictions.extend(predicted_indices.cpu().numpy().tolist())
            
            # Create sequence IDs
            batch_size = seq_embeddings.shape[0]
            batch_ids = [len(sequence_ids) + j for j in range(batch_size)]
            sequence_ids.extend(batch_ids)
    
    embeddings = np.vstack(embeddings)
    
    if max_sequences:
        embeddings = embeddings[:max_sequences]
        sequence_ids = sequence_ids[:max_sequences]
        predictions = predictions[:max_sequences]
    
    print(f"Extracted {embeddings.shape[0]} sequence embeddings of dimension {embeddings.shape[1]}")
    print(f"Collected {len(predictions)} predictions")
    return embeddings, sequence_ids, predictions

def get_prediction_metadata_per_cluster(
    predictions_per_cluster: Dict[int, List[int]],
    item_meta_dict: Dict[int, Dict],
    id2item: Dict[int, str]
) -> Dict[int, List[Dict]]:
    """
    Get metadata for predicted items in each cluster.
    
    Args:
        predictions_per_cluster: dict {cluster_label: [predicted_item1, predicted_item2, ...]}
        item_meta_dict: dict mapping item names to metadata
        id2item: dict mapping item IDs to item names
    
    Returns:
        prediction_metadata_per_cluster: dict {cluster_label: [meta1, meta2, ...]}
    """
    prediction_metadata_per_cluster = {}
    for cluster_label, predicted_items in predictions_per_cluster.items():
        cluster_metadata = []
        for item_id in predicted_items:
            item_name = id2item[item_id]
            item_meta = item_meta_dict[item_name]
            cluster_metadata.append(item_meta)
        prediction_metadata_per_cluster[cluster_label] = cluster_metadata
    return prediction_metadata_per_cluster


def get_predictions_per_cluster(clustering_results: Dict[int, Dict], prediction_ids: List[int], k: int) -> Dict[int, List[int]]:
    if k not in clustering_results:
        raise ValueError(f"K={k} not found in clustering results")

    labels = clustering_results[k]['labels']
    cluster_to_predictions = {}

    for label in np.unique(labels):
        cluster_to_predictions[label] = [seq_id for seq_id, lbl in zip(prediction_ids, labels) if lbl == label]

    return cluster_to_predictions

def cluster_sequences_with_predictions(model, dataloader, device, k_min=2, k_max=20, 
                                     output_dir='clustering_results', visualize=True):
    """
    Main function to perform clustering analysis on sequence embeddings and collect predictions
    
    Args:
        model: RecformerForSeqRec model
        dataloader: DataLoader containing test data
        device: Device to run model on
        k_min: Minimum number of clusters
        k_max: Maximum number of clusters
        output_dir: Directory to save results
        visualize: Whether to create visualizations
    
    Returns:
        clustering_results: Dictionary containing clustering results for each k
        embeddings: Extracted embeddings
        sequence_ids: Sequence identifiers
        predictions: Predicted items for each sequence
        predictions_per_cluster: Predictions per cluster
    """
    
    print("Starting clustering analysis with predictions...")
    print("="*50)

    os.makedirs(output_dir, exist_ok=True)
    
    # Extract embeddings and predictions
    embeddings, sequence_ids, predictions = extract_embeddings_with_predictions(model, dataloader, device)
    print(f"Extracted {embeddings.shape[0]} embeddings and {len(predictions)} predictions")
    
    # Perform K-means clustering with different k values
    k_range = range(k_min, k_max + 1)
    clustering_results = perform_kmeans_clustering(embeddings, k_range)
    
    # Analyze results
    optimal_k = analyze_clustering_results(clustering_results, output_dir=output_dir)

    # Get sequence IDs and data points per cluster
    predictions_per_cluster = get_predictions_per_cluster(clustering_results, predictions, optimal_k)
        
    # Save results
    save_clustering_results(clustering_results, sequence_ids, output_dir)
    print(f"Clustering results saved to {output_dir}")

    # Save data points
    predictions_per_cluster = {int(k): v for k, v in predictions_per_cluster.items()}
    with open(os.path.join(output_dir, "item_pred_per_cluster_label.json"), 'w') as f:
        json.dump(predictions_per_cluster, f, indent=2)
    
    # Save predictions per cluster
    predictions_per_cluster = {int(k): v for k, v in predictions_per_cluster.items()}
    with open(os.path.join(output_dir, "predictions_per_cluster.json"), 'w') as f:
        json.dump(predictions_per_cluster, f, indent=2)
    
    # Visualize clusters for optimal k
    if visualize:
        print(f"Visualizing clusters for optimal k={optimal_k}")
        optimal_labels = clustering_results[optimal_k]['labels']
        visualize_clusters(embeddings, optimal_labels, optimal_k, method='tsne', output_dir=output_dir)
        visualize_clusters(embeddings, optimal_labels, optimal_k, method='pca', output_dir=output_dir)

    # Print detailed statistics for optimal k
    get_cluster_statistics(clustering_results, optimal_k)
    
    return (clustering_results, embeddings, sequence_ids, predictions, predictions_per_cluster)


# Modified main function
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

    # data process
    parser.add_argument('--preprocessing_num_workers', type=int, default=8, help="The number of processes to use for the preprocessing.")
    parser.add_argument('--dataloader_num_workers', type=int, default=0)

    # eval
    parser.add_argument('--num_train_epochs', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--finetune_negative_sample_size', type=int, default=1000)
    parser.add_argument('--metric_ks', nargs='+', type=int, default=[10, 50], help='ks for Metric@k')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fix_word_embedding', action='store_true')
    parser.add_argument('--verbose', type=int, default=3)

    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.device)) if args.device>=0 else torch.device('cpu')

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
                            shuffle=False,
                            batch_size=args.batch_size, 
                            collate_fn=test_data.collate_fn)
    
    if args.fix_word_embedding:
        print('Fix word embeddings.')
        for param in model.longformer.embeddings.word_embeddings.parameters():
            param.requires_grad = False

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
    
    # Replace the original clustering call with:
    (clustering_results, embeddings, sequence_ids, predictions, predictions_per_cluster) = cluster_sequences_with_predictions(
        model=model,
        dataloader=test_loader,
        device=args.device, 
        k_min=2,
        k_max=4,
        output_dir='clustering_results',
        visualize=True
    )

    print("Clustering results:", clustering_results)
    print("Predictions per cluster:", predictions_per_cluster)


    # Get metadata for predicted items (new functionality)
    prediction_metadata_per_cluster = get_prediction_metadata_per_cluster(
        predictions_per_cluster,
        item_meta_dict,
        id2item
    )

    prediction_metadata_per_cluster = {int(k): v for k, v in prediction_metadata_per_cluster.items()}
    with open(os.path.join("clustering_results", "prediction_metadata_per_cluster.json"), 'w') as f:
        json.dump(prediction_metadata_per_cluster, f, indent=2)

    # Generate cluster descriptions for predictions
    prediction_cluster_descriptions = {}
    
    for k in prediction_metadata_per_cluster.keys():
        # Description based on predicted items
        predicted_sequences = prediction_metadata_per_cluster[k]
        prediction_cluster_descriptions[k] = get_cluster_description(predicted_sequences)
        print(f"Cluster {k} (predictions) description: {prediction_cluster_descriptions[k]}")

    # Save descriptions        
    prediction_cluster_descriptions = {int(k): v for k, v in prediction_cluster_descriptions.items()}
    with open(os.path.join("clustering_results", "prediction_cluster_descriptions.json"), 'w') as f:
        json.dump(prediction_cluster_descriptions, f, indent=2)

if __name__ == "__main__":
    main()