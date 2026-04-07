#!/usr/bin/env python3
"""
Material Prediction Analysis
转换自 analysis.ipynb
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
import contriever.src.contriever
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

print("✓ 所有库导入完成！\n")


def embed_queries(queries, model, tokenizer):
    device = 'cuda:0'
    model.eval()
    embeddings, batch_question = [], []
    with torch.no_grad():

        for k, q in tqdm(enumerate(queries)):
            # q = q.lower()
            # q = contriever.src.normalize_text.normalize(q)

            batch_question.append(q)

            encoded_batch = tokenizer.batch_encode_plus(
                batch_question,
                return_tensors="pt",
                max_length=512,
                padding=True,
                truncation=True,
            )

            encoded_batch = {k: v.to(device) for k, v in encoded_batch.items()}
            output = model(**encoded_batch)
            embeddings.append(output.cpu())

            batch_question = []

    embeddings = torch.cat(embeddings, dim=0).numpy()

    print(f"Questions embeddings shape: {embeddings.shape}")

    return embeddings


def get_test_embedding(description, normalize_embeddings):
    test_embedding = model.encode(description, normalize_embeddings=normalize_embeddings)
    return test_embedding


def get_train_embedding(fname, key_name, normalize_embeddings, model_name):
    with open(fname) as f:
        data = json.load(f)

    data = data[key_name]
    train_embedding = {}
    train_mean_embedding = {}
    for key, values in data.items():
        # if key.find('high') != -1:
        #     values = ['high'] * 10
        # else:
        #     values = ['low'] * 10

        if model_name == 'OpenScholar_Reranker':
            train_embedding[key] = model.encode(values, normalize_embeddings=normalize_embeddings)
        else:
            train_embedding[key] = embed_queries_internal(values, query_encoder, query_tokenizer)

        train_mean_embedding[key] = train_embedding[key].mean(axis=0, keepdims=True)
    return train_embedding, train_mean_embedding


def embed_queries_internal(queries, model, tokenizer):
    """内部使用的embed_queries，不显示进度条"""
    device = 'cuda:0'
    model.eval()
    embeddings, batch_question = [], []
    with torch.no_grad():

        for k, q in enumerate(queries):
            # q = q.lower()
            # q = contriever.src.normalize_text.normalize(q)

            batch_question.append(q)

            encoded_batch = tokenizer.batch_encode_plus(
                batch_question,
                return_tensors="pt",
                max_length=512,
                padding=True,
                truncation=True,
            )

            encoded_batch = {k: v.to(device) for k, v in encoded_batch.items()}
            output = model(**encoded_batch)
            embeddings.append(output.cpu())

            batch_question = []

    embeddings = torch.cat(embeddings, dim=0).numpy()

    return embeddings


# ============================================================
# Cell 1: 加载模型和训练数据
# ============================================================
print("正在加载模型...")
# model_name = 'OpenScholar_Reranker'
model_name = 'contriever'

if model_name == 'OpenScholar_Reranker':
    model = SentenceTransformer('/data/oyyw/llm/OpenScholar_Reranker')
    model = model.to('cuda:0')
    normalize_embeddings = True
    model.eval()
else:
    query_encoder, query_tokenizer, _ = contriever.src.contriever.load_retriever('akariasai/pes2o_contriever')
    query_encoder = query_encoder.to('cuda:0')
    normalize_embeddings = True
    query_encoder.eval()

print(f"✓ 模型加载完成 ({model_name}) - 已移至 CUDA\n")

fname = 'merged.json'
key_name = 'all_description'

print(f"正在加载训练数据: {fname}")
train_embedding, train_mean_embedding = get_train_embedding(fname, key_name, normalize_embeddings, model_name)
print("✓ 训练数据加载完成\n")


# ============================================================
# Cell 2: 准备测试数据
# ============================================================
print("正在准备测试数据...")
chda = "Trans-1,4-diaminocyclohexane is a cyclohexane derivative with two amino groups (-NH₂) attached at the 1st and 4th positions of the cyclohexane ring. In its trans configuration, the amino groups are positioned on opposite sides of the ring, minimizing steric clashes. This trans arrangement is crucial for the molecule's stability, allowing favorable spatial orientation for subsequent interactions."
hda = "1,6-hexamethylenediamine (C₆H₁₄N₂) is a linear organic compound with two amino groups positioned at opposite ends of a flexible six-carbon chain. The molecule’s flexibility arises from single bonds between carbons, allowing rotational freedom. Upon protonation in acidic solution, the -NH₂ groups become -NH₃⁺, enhancing hydrogen bonding capabilities."

materials = [chda, hda]
if model_name == 'OpenScholar_Reranker':
    test_embedding = get_test_embedding(materials, normalize_embeddings=normalize_embeddings)
else:
    test_embedding = embed_queries_internal(materials, query_encoder, query_tokenizer)

print("✓ 测试数据准备完成\n")


# ============================================================
# Cell 3: 计算相似度
# ============================================================
print("正在计算相似度...")
for i in ['stiffness', 'carrier mobility', 'band gap', 'sensitivity in X-ray detection', 'Ion migration activation energy of the crystal']:
    for j in ['high', ]:
        property = f'{j} {i}'
        sim = cosine_similarity(test_embedding, train_embedding[property])
        top10_indices = np.argsort(sim, axis=1)[:, -10:]
        sim = sim[np.arange(2)[:, None], top10_indices]
        print(f"{property}: {np.array(sim).mean(axis=-1)}")

print()


# ============================================================
# Cell 4: KNN分析和绘图
# ============================================================
print("正在运行KNN分析并生成图表...")

for i in ['stiffness', 'carrier mobility', 'band gap', 'sensitivity in X-ray detection', 'Ion migration activation energy of the crystal']:
    print(f"  处理属性: {i}")

    all_high_preds = []
    all_neighbors = []

    for n_neighbors in range(20, 100, 5):
        all_neighbors.append(n_neighbors)

        knn = KNeighborsClassifier(n_neighbors=n_neighbors)

        x_train = np.concatenate([train_embedding[f'high {i}'], train_embedding[f'low {i}']], axis=0)
        y_train = np.array(len(train_embedding[f'high {i}']) * [1] + len(train_embedding[f'low {i}']) * [0])
        x_test = test_embedding

        knn.fit(x_train, y_train)
        y_pred = knn.predict_proba(x_test)
        high_pred = y_pred[range(2), 1]
        all_high_preds.append(high_pred)

    # Extract the first and second elements from each array
    first_elements = [pred[0] for pred in all_high_preds]
    second_elements = [pred[1] for pred in all_high_preds]

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot first elements
    plt.plot(all_neighbors, first_elements, label='CHDA', marker='o', linestyle='-', color='blue')

    # Plot second elements
    plt.plot(all_neighbors, second_elements, label='HDA', marker='x', linestyle='--', color='orange')

    # Adding labels and title
    plt.xlabel('k')
    plt.ylabel('Prediction Probability')
    upper_property = ' '.join([tmp.title() if tmp not in ['the', 'of', 'in'] else tmp \
                                for tmp in i.split(' ')])
    plt.title(f'Prediction Probabilities for High {upper_property}')

    plt.legend()

    # Save the plot
    output_filename = f'output/{i}.png'
    plt.savefig(output_filename)
    print(f"    ✓ 图表已保存: {output_filename}")
    plt.close()

print("\n✓ KNN分析完成\n")

