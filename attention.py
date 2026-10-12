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