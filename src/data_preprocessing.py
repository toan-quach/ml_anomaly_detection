import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder


def remove_duplicates(data):
    success_non_dup_data = data[data["Label"] == "Success"].drop_duplicates(subset=['Features'])
    fail_non_dup_data = data[data["Label"] == "Fail"].drop_duplicates(subset=['Features'])
    data = pd.merge(success_non_dup_data, fail_non_dup_data, how="outer").reset_index(drop=True)
    return data


def drop_success(data, dropSuccessFrac):
    success_idx = (
        data.loc[data["Label"] == "Success"]
        .sample(frac=dropSuccessFrac, random_state=42)
        .index
    )
    data = data.drop(success_idx).reset_index(drop=True)
    return data


def one_hot_encoding(data):
    # OneHotEncoder
    unique_features = data["Features"].map(lambda x: [i[1:] for i in x]).explode().unique().astype(np.int32)
    unique_features.sort()
    # encode features
    encoder = OneHotEncoder(sparse_output=False, dtype=np.float32)
    encoded_features = encoder.fit(unique_features.reshape(-1, 1))
    data["Features_Encoded"] = data["Features"].map(lambda x: encoder.transform([[int(i[1:])] for i in x]))

    # clean data to reduce memory usage
    data = data[["Features_Encoded", "TimeInterval", "Label"]].copy()

    # Maximum length of sequence is chosen as 50 based on the distribution of sequence lengths
    # Sequences beyond this length are rare and has drippled effect on buiding even sequences
    # (i.e., padding/truncating) for training. Most short or medium sequences from 1 -> 40 in
    # length are padded up to 200s if this maximum length is not chosen.
    MAX_LEN = 50  # maximum sequence length
    ENCODED_DIMEN = data["Features_Encoded"].iloc[0].shape[1]  # for 'one-hot' encoding

    # pad with zeros on the left to create even sequences for training
    def left_pad_feature(seq, pad_len=MAX_LEN, encoded_dim=ENCODED_DIMEN):
        seq = np.stack(seq)  # shape: (L, 384)
        L = seq.shape[0]
        if L >= pad_len:
            return seq[-pad_len:]  # truncate if too long
        pad = np.zeros((pad_len - L, encoded_dim), dtype=seq.dtype)
        return np.vstack([pad, seq])

    data["Features_Encoded_Padded"] = data["Features_Encoded"].map(left_pad_feature)
    data = data[["Features_Encoded_Padded", "TimeInterval", "Label"]]
    data["Features_Encoded_Padded"] = data["Features_Encoded_Padded"].map(lambda x: torch.from_numpy(x))
    return data


def minilm_embedding(data):
    # Text embedding using Sentence-BERT
    from sentence_transformers import SentenceTransformer
    template_data = pd.read_csv("../../data/preprocessed/HDFS.log_templates.csv")
    model_name = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)
    embeddings = model.encode(template_data['EventTemplate'].tolist())
    template_embedding_dict = {template_id: template_embedding for template_id, template_embedding in
                               zip(template_data["EventId"].tolist(), embeddings)}
    data["Features_Embedded"] = data["Features"].map(lambda x: [template_embedding_dict[i] for i in x])

    # clean data to reduce memory usage
    data = data[["Features_Embedded", "TimeInterval", "Label"]].copy()

    # Maximum length of sequence is chosen as 50 based on the distribution of sequence lengths
    # Sequences beyond this length are rare and has drippled effect on buiding even sequences
    # (i.e., padding/truncating) for training. Most short or medium sequences from 1 -> 40 in
    # length are padded up to 200s if this maximum length is not chosen.
    MAX_LEN = 50  # maximum sequence length
    EMBED_DIM = 384  # for 'all-MiniLM-L6-v2'

    # pad with zeros on the left to create even sequences for training
    def left_pad_feature(seq, pad_len=MAX_LEN, embed_dim=EMBED_DIM):
        seq = np.stack(seq)  # shape: (L, 384)
        L = seq.shape[0]
        if L >= pad_len:
            return seq[-pad_len:]  # truncate if too long
        pad = np.zeros((pad_len - L, embed_dim), dtype=seq.dtype)
        return np.vstack([pad, seq])

    data["Features_Embedded_Padded"] = data["Features_Embedded"].map(left_pad_feature)
    data = data[["Features_Embedded_Padded", "TimeInterval", "Label"]]
    data["Features_Embedded_Padded"] = data["Features_Embedded_Padded"].map(lambda x: torch.from_numpy(x))
    return data


def preprocess_data(embedding="onehot", removeDuplicates=True, dropSuccess=False, dropSuccessFrac=0.40):
    data = pd.read_csv("../../data/preprocessed/Event_traces.csv")

    # remove duplicates
    if removeDuplicates:
        data = remove_duplicates(data)

    # Drop dropSuccessFrac% of 'Success' rows to reduce memory consumption
    if dropSuccess:
        data = drop_success(data, dropSuccessFrac)

    # convert string sequence to list sequence
    data["Features"] = data["Features"].map(lambda x: x[1:-1].split(","))
    # can map timeinterval with the operation ran
    data["TimeInterval"] = data["TimeInterval"].map(lambda x: [float(i) for i in x[1:-1].split(",")])

    if embedding == "onehot":
        data = one_hot_encoding(data)
    elif embedding == "minilm":
        data = minilm_embedding(data)
    else:
        raise ValueError(
            f"Unknown embedding_method: {embedding}. "
            "Choose from ['onehot', 'minilm']"
        )

    label_encoder = OneHotEncoder(sparse_output=False, dtype=np.float32)
    encoded_labels = label_encoder.fit_transform(data["Label"].values.reshape(-1, 1))

    return data, label_encoder, encoded_labels
