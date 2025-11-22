# Methodology

To detect anomalous system behavior effectively, we implemented two sequence classification models: a Long Short-Term Memory (LSTM) network as a temporal baseline and a Transformer-based encoder to leverage self-attention mechanisms. The selection of these architectures allows us to evaluate the trade-off between strictly sequential processing and global context modeling in the domain of system call analysis.

## Data Representation and Preprocessing

The raw input data consists of variable-length sequences of system calls (e.g., `aue_open`, `aue_read`), which represent the execution traces of processes. To transform these discrete events into a format suitable for neural networks, we adopted a vocabulary-based tokenization strategy. Each unique system call type corresponds to a specific integer identifier, with reserved tokens for padding (`<PAD>`) and unknown events (`<UNK>`). 

Input sequences were standardized to a fixed length of 512 tokens. This length was chosen to cover significant portions of execution traces while maintaining computational efficiency. To handle the substantial class imbalance inherent in intrusion detection—where "normal" traffic vastly outweighs "attack" samples—we incorporated a weighted random sampling strategy during training. Furthermore, the binary cross-entropy loss function was weighted to penalize misclassifications of the minority class more heavily, ensuring the model learns robust features for attack detection rather than biasing towards the majority class.

## Architectural Design

Both models share a unified input dimension ($d_{model} = 64$) to ensure a fair comparison. This embedding layer projects discrete token IDs into a dense, continuous vector space, enabling the models to learn semantic relationships between different system calls.

### Long Short-Term Memory (LSTM) Baseline
We selected a unidirectional LSTM as our baseline architecture due to its established effectiveness in modeling temporal dependencies. System calls are inherently sequential; the validity of an operation often depends on the state established by preceding calls (e.g., a `read` must follow an `open`). The LSTM processes the sequence token-by-token, maintaining a hidden state that acts as a memory of the sequence history.

To prevent information loss in long sequences—a common issue where the final hidden state "forgets" early events—we implemented a **masked mean pooling** mechanism. Instead of relying solely on the last hidden state, the model calculates the average of the hidden states across all valid (non-padding) time steps. This aggregates information from the entire execution trace into a single fixed-size vector, providing a more global representation of the process behavior before classification.

### Transformer Encoder
To address the limitations of sequential processing and better capture complex, long-range dependencies, we implemented a Transformer-based classifier. Unlike the LSTM, the Transformer processes the entire sequence in parallel using multi-head self-attention. This mechanism allows the model to dynamically weigh the importance of different system calls relative to each other, regardless of their distance in the sequence. For instance, a specific file access pattern might only be anomalous when viewed in the context of a network connection that occurred hundreds of steps earlier.

Since the self-attention mechanism is permutation-invariant, we augmented the token embeddings with learnable **positional embeddings** to inject necessary order information. The architecture consists of two encoder layers with four attention heads, balancing model capacity with the relatively small vocabulary size of system calls. Similar to the LSTM, the output of the Transformer encoder undergoes masked mean pooling to generate a comprehensive sequence embedding for the final binary classification layer.

## Training and Evaluation
Both models were trained using the Adam optimizer with a learning rate of $1\times10^{-3}$ for three epochs. We utilized a class-weighted Binary Cross Entropy with Logits loss to directly optimize for the skewed distribution. To rigorously assess the detection capabilities, we evaluated the models using a suite of metrics including Precision, Recall, F1-score, and ROC-AUC, prioritizing the identification of attacks (Recall) while minimizing false alarms (Precision).
