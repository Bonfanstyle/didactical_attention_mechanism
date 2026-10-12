import numpy as np

def softmax(X):
    """
    Compute the softmax of a numpy array.

    Args:
        X: A numpy array of shape (batch_size, seq_len_q, seq_len_k)
    """
    # Mathematically speaking, the exp(-max(X)) is not necessary because it simplifies. But in practice, to avoid nan caused by exp(large number), the max is substracted.
    e_Xn = np.exp(X - np.max(X, axis=-1, keepdims=True))
    return e_Xn / np.sum(e_Xn, axis=-1, keepdims=True) # To have a sum by line, we sum on the columns indexes, so we apply the sum on the last axis.

def attention(Q, K, V, causal = False):
    """
    Compute the attention output.

    Args:
        Q: A numpy array of shape (batch_size, seq_len_q, d_k)
        K: A numpy array of shape (batch_size, seq_len_k, d_k)
        V: A numpy array of shape (batch_size, seq_len_k, d_v)
        causal: A boolean indicating whether to apply causal masking
    """
    if causal and Q.shape[-2] != K.shape[-2]:
        raise ValueError("Causal attention requires seq_len_q to be equal to seq_len_k.")
    scaled_scores = Q @ K.mT / np.sqrt(K.shape[-1]) # .mT transpose only for the last two dimensions.
    if causal:
        mask = np.triu(np.full(scaled_scores.shape[-2:], -np.inf), k=1) # np.triu creates an upper triangular matrix. In softmax we apply exp function, so -inf will become 0 and will be ignored.
        scaled_scores += mask # Just adding here because 0 + score = score, and -inf + score = -inf, which will be ignored by softmax.
        # Broadcasting rules will apply mask to the two last dimensions of scaled_scores, which mean on all the batch.
    attention_output = np.matmul(softmax(scaled_scores), V)
    return attention_output

def apply_self_attention(X, W_q, W_k, W_v, causal = False):
    """
    Apply self-attention mechanism to the input data.

    Args:
        X: A numpy array of shape (batch_size, seq_len, d_model)
        W_q: A numpy array of shape (d_model, d_k)
        W_k: A numpy array of shape (d_model, d_k)
        W_v: A numpy array of shape (d_model, d_v)
    """
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return attention(Q, K, V, causal=causal)

def apply_cross_attention(X_q, X_kv, W_q, W_k, W_v):
    """
    Apply cross-attention mechanism to the input data.

    Args:
        X_q: A numpy array of shape (batch_size, seq_len_q, d_model)
        X_kv: A numpy array of shape (batch_size, seq_len_kv, d_model)
        W_q: A numpy array of shape (d_model, d_k)
        W_k: A numpy array of shape (d_model, d_k)
        W_v: A numpy array of shape (d_model, d_v)
    """
    Q = X_q @ W_q
    K = X_kv @ W_k
    V = X_kv @ W_v
    return attention(Q, K, V, causal=False)

def apply_multi_head_attention(X_q, X_kv, W_q, W_k, W_v, num_heads, W_o, causal=False):
    """
    Apply multi-head attention mechanism to the input data.

    Args:
        X_q: A numpy array of shape (batch_size, seq_len_q, d_model)
        X_kv: A numpy array of shape (batch_size, seq_len_kv, d_model)
        W_q: A numpy array of shape (d_model, d_k * num_heads)
        W_k: A numpy array of shape (d_model, d_k * num_heads)
        W_v: A numpy array of shape (d_model, d_v * num_heads)
        num_heads: An integer representing the number of attention heads
        W_o: A numpy array of shape (d_v * num_heads, d_model)
        causal: A boolean indicating whether to apply causal masking
    """
    batch_size = X_q.shape[0]
    seq_len_q = X_q.shape[1]
    seq_len_kv = X_kv.shape[1]
    d_k = W_q.shape[1] // num_heads
    d_v = W_v.shape[1] // num_heads

    # Reshape and transpose for multi-head attention
    Q = (X_q @ W_q).reshape(batch_size, seq_len_q, num_heads, d_k).transpose(0, 2, 1, 3)  # (batch_size, num_heads, seq_len_q, d_k)
    K = (X_kv @ W_k).reshape(batch_size, seq_len_kv, num_heads, d_k).transpose(0, 2, 1, 3)  # (batch_size, num_heads, seq_len_kv, d_k)
    V = (X_kv @ W_v).reshape(batch_size, seq_len_kv, num_heads, d_v).transpose(0, 2, 1, 3)  # (batch_size, num_heads, seq_len_kv, d_v)

    # Compute attention for each head
    attention_outputs = attention(Q, K, V, causal=causal)  # (batch_size, num_heads, seq_len_q, d_v). We can use the attention function directly because it supports batch processing and multiple heads, due to the usage of the last two dimensions for the dot product and softmax.
    # Applying the inverse transformations before product with W_o
    attention_outputs = attention_outputs.transpose(0, 2, 1, 3).reshape(batch_size, seq_len_q, num_heads * d_v)  # (batch_size, seq_len_q, d_v * num_heads)

    return attention_outputs @ W_o  # (batch_size, seq_len_q, d_model)