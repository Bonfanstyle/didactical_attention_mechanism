import attention
import numpy as np

import pytest

# NOTE : This test file is designed to check properties of the attention mechanism, rather than be an exhaustive unit test suite.

# Global parameters for the tests
batch_size = 16
seq_len_q = 10
seq_len_k = 15
d_k = 64
d_v = 32
d_model = 128

@pytest.fixture
def rng():
    return np.random.default_rng(0)  # Use a fixed seed for reproducibility

@pytest.fixture
def random_arrays(rng):
    """
    Fixture to generate random arrays for X, W_q, W_k, and W_v.
    """
    X = rng.normal(size=(batch_size, seq_len_q, d_model))
    W_q = rng.normal(size=(d_model, d_k)) / np.sqrt(d_model)
    W_k = rng.normal(size=(d_model, d_k)) / np.sqrt(d_model)
    W_v = rng.normal(size=(d_model, d_v)) / np.sqrt(d_model)
    return X, W_q, W_k, W_v

@pytest.fixture
def self_attention_arrays(random_arrays):
    """
    Fixture to generate Q, K, V arrays from X and weight matrices.
    """
    X, W_q, W_k, W_v = random_arrays
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return Q, K, V

@pytest.fixture
def cross_attention_arrays(rng):
    """
    Fixture to generate random arrays for cross-attention tests.
    """
    X_q = rng.normal(size=(batch_size, seq_len_q, d_model))
    X_kv = rng.normal(size=(batch_size, seq_len_k, d_model))
    W_q = rng.normal(size=(d_model, d_k)) / np.sqrt(d_model)
    W_k = rng.normal(size=(d_model, d_k)) / np.sqrt(d_model)
    W_v = rng.normal(size=(d_model, d_v)) / np.sqrt(d_model)
    return X_q, X_kv, W_q, W_k, W_v

@pytest.fixture
def multi_head_attention_arrays(rng):
    """
    Fixture to generate random arrays for multi-head attention tests.
    """
    num_heads = 4
    d_k_multi = d_k * num_heads
    d_v_multi = d_v * num_heads
    X_q = rng.normal(size=(batch_size, seq_len_q, d_model))
    X_kv = rng.normal(size=(batch_size, seq_len_k, d_model))
    W_q = rng.normal(size=(d_model, d_k_multi)) / np.sqrt(d_model)
    W_k = rng.normal(size=(d_model, d_k_multi)) / np.sqrt(d_model)
    W_v = rng.normal(size=(d_model, d_v_multi)) / np.sqrt(d_model)
    W_o = rng.normal(size=(d_v_multi, d_model)) / np.sqrt(d_v_multi)
    return X_q, X_kv, W_q, W_k, W_v, num_heads, W_o


def test_softmax(rng):
    X = rng.normal(size=(batch_size, seq_len_q, seq_len_k))
    X_large = X*1000  # Scale up the values to test numerical stability
    res = attention.softmax(X)
    res_large = attention.softmax(X_large)
    np.testing.assert_allclose(np.sum(res, axis=-1), 1.0)  # Check if the sum of softmax outputs is 1 for each row
    np.testing.assert_allclose(np.sum(res_large, axis=-1), 1.0)
    hard_X = np.array([[1000, 1000, 1000], [1, 2, 3]])
    res_hard = attention.softmax(hard_X)
    hard_expected = np.array([[1/3, 1/3, 1/3], [0.09003057, 0.24472847, 0.66524096]])
    np.testing.assert_allclose(res_hard, hard_expected)

def test_self_attention_shapes(self_attention_arrays):
    out = attention.attention(*self_attention_arrays)
    assert out.shape == (batch_size, seq_len_q, d_v)

def test_causal_first_token_sees_itself(self_attention_arrays):
    Q, K, V = self_attention_arrays
    out_causal = attention.attention(Q, K, V, causal=True)
    # softmax(Q @ K.mT / np.sqrt(K.shape[-1]))[:, 0, 0] should be 1 when causal mask is applied. The first token only sees itself.
    np.testing.assert_allclose(out_causal[:, 0, :], V[:, 0, :])

def test_causal_ignore_future(rng, random_arrays):
    X, W_q, W_k, W_v = random_arrays
    out = attention.apply_self_attention(X, W_q, W_k, W_v, causal=True)
    X[:, -1, :] = rng.normal(size=(batch_size, d_model)) # Change the last token in the sequence to see if only the last token is affected by the causal masking.
    out_modified = attention.apply_self_attention(X, W_q, W_k, W_v, causal=True)
    # The outputs for the first seq_len_q-1 tokens should be the same when the last token is modified.
    np.testing.assert_allclose(out[:, :-1, :], out_modified[:, :-1, :])
    # The output for the last token should be different when the last token is modified.
    assert not np.allclose(out[:, -1, :], out_modified[:, -1, :])

def test_cross_attention_shapes(cross_attention_arrays):
    out = attention.apply_cross_attention(*cross_attention_arrays)
    assert out.shape == (batch_size, seq_len_q, d_v)

def test_cross_attention_modify_query(rng, cross_attention_arrays):
    X_q, X_kv, W_q, W_k, W_v = cross_attention_arrays
    out = attention.apply_cross_attention(X_q, X_kv, W_q, W_k, W_v)
    token_idx = rng.integers(0, seq_len_q)
    X_q[:, token_idx, :] = rng.normal(size=(batch_size, d_model))
    out_modified = attention.apply_cross_attention(X_q, X_kv, W_q, W_k, W_v)
    boolean_mask = np.arange(seq_len_q) != token_idx
    # The output for all tokens should be the same except for the modified one.
    np.testing.assert_allclose(out[:, boolean_mask, :], out_modified[:, boolean_mask, :])
    # The output for the modified token should be different when the query token is modified.
    assert not np.allclose(out[:, token_idx, :], out_modified[:, token_idx, :])

def test_cross_attention_modify_key_value(rng, cross_attention_arrays):
    X_q, X_kv, W_q, W_k, W_v = cross_attention_arrays
    out = attention.apply_cross_attention(X_q, X_kv, W_q, W_k, W_v)
    token_idx = rng.integers(0, seq_len_k)
    X_kv[:, token_idx, :] = rng.normal(size=(batch_size, d_model))
    out_modified = attention.apply_cross_attention(X_q, X_kv, W_q, W_k, W_v)
    # Every row output should be different when the key/value token is modified.
    assert not np.isclose(out, out_modified).all(axis=-1).any()

def test_multi_head_attention_shapes(multi_head_attention_arrays):
    X_q, X_kv, W_q, W_k, W_v, num_heads, W_o = multi_head_attention_arrays
    out = attention.apply_multi_head_attention(X_q, X_kv, W_q, W_k, W_v, num_heads, W_o)
    assert out.shape == (batch_size, seq_len_q, d_model)

def test_multi_head_attention_modify_head_column(rng, multi_head_attention_arrays):
    X_q, X_kv, W_q, W_k, W_v, num_heads, W_o = multi_head_attention_arrays
    I = np.eye(d_v * num_heads)  # Identity matrix for the output projection, in order to check concatenation. Auto broadcasted to batch size with @ operator in apply_multi_head_attention.
    out = attention.apply_multi_head_attention(X_q, X_kv, W_q, W_k, W_v, num_heads, I)
    head_idx = rng.integers(0, num_heads)
    col_idx = rng.integers(0, d_k)
    # Modify a specific column in the weight matrix for one head
    W_q[:, head_idx * d_k + col_idx] = rng.normal(size=(d_model,))
    out_modified = attention.apply_multi_head_attention(X_q, X_kv, W_q, W_k, W_v, num_heads, I)
    # The concateneted output (before the product with W_o) for the modified head should be different when the weight matrix is modified.
    assert not np.allclose(out[:, :, head_idx * d_v:(head_idx + 1) * d_v], out_modified[:, :, head_idx * d_v:(head_idx + 1) * d_v]).all(axis=-1).any() # Same mechanism as line 126.
    boolean_mask = np.repeat(np.arange(num_heads) != head_idx, d_v)  # Repeat the boolean mask for each column in the head's output
    # The concatenated output (before the product with W_o) for all other heads should be the same when the weight matrix for one head is modified.
    np.testing.assert_allclose(out[:, :, boolean_mask], out_modified[:, :, boolean_mask])

def test_multi_head_attention_equivalence_to_cross_attention(rng, multi_head_attention_arrays):
    X_q, X_kv, W_q, W_k, W_v, num_heads, W_o = multi_head_attention_arrays
    # For each head, the multi-head attention should be equivalent to the cross-attention.
    # Compute the output of cross attention for each head and concatenate the results
    concatenated_cross_attention_out = []
    for head_idx in range(num_heads):
        single_head_W_q = W_q[:, head_idx * d_k:(head_idx + 1) * d_k]
        single_head_W_k = W_k[:, head_idx * d_k:(head_idx + 1) * d_k]
        single_head_W_v = W_v[:, head_idx * d_v:(head_idx + 1) * d_v]
        out_single_head = attention.apply_cross_attention(X_q, X_kv, single_head_W_q, single_head_W_k, single_head_W_v)
        concatenated_cross_attention_out.append(out_single_head)
    concatenated_cross_attention_out = np.concatenate(concatenated_cross_attention_out, axis=-1)
    out_multi_head = attention.apply_multi_head_attention(X_q, X_kv, W_q, W_k, W_v, num_heads, np.eye(d_v * num_heads))
    np.testing.assert_allclose(concatenated_cross_attention_out, out_multi_head)