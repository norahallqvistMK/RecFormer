from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

def train_and_save_decision_tree(cluster_data, max_features=50, max_depth=4, filename="tree.png"):
    texts = []
    labels = []

    # Step 1: Convert item lists into text samples and labels
    for cluster_label, users in cluster_data.items():
        for user_items in users:
            combined_text = " ".join(
                f"{item.get('title', '')} {item.get('brand', '')} {item.get('category', '')}"
                for item in user_items
            )
            texts.append(combined_text)
            labels.append(int(cluster_label))  # convert label string to int

    # Step 2: TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    # Step 3: Train a decision tree
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X, labels)

    # Step 4: Save the tree plot to file
    plt.figure(figsize=(20, 10))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=[f"Cluster {i}" for i in sorted(set(labels))],
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title("Decision Tree Explaining Cluster Assignments")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()  # prevent display in some environments

# cluster_data = {
#     "0": [
#         [{"title": "Item A1", "brand": "Brand X", "description": "Description of item A1"}],
#         [{"title": "Item A2", "brand": "Brand Y", "description": "Description of item A2"}]
#     ],
#     "1": [
#         [{"title": "Item B1", "brand": "Brand Z", "description": "Description of item B1"}],
#         [{"title": "Item B2", "brand": "Brand W", "description": "Description of item B2"}]
#     ]
# }

# train_and_save_decision_tree(cluster_data, filename="cluster_tree.png")
