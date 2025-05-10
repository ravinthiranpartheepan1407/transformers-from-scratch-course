#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Install dependencies

# Python IDE: #pip install numpy matplotlib seaborn tqdm
# Anaconda Nav: conda install numpy matplotlib seaborn tqdm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import copy


# In[2]:


# Class - Where you can your data and behaviors (methods)
class Transformer:
    # vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dropout 
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        # self
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_prob = dropout
        
        # Implement world level embeddings 
        # Create random weights for embedding:
        # Word : [0.1, 02]
        # Initializes the word embedding with values sampled from a normal distribution or Gaussian distribution
        # Xavier Style Initialization
        self.embedding = np.random.randn(vocab_size, d_model) * np.sqrt(1.0/d_model)
        
        # Initialize Encoders and Decoders
        self.encoder_layers = [
            EncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ]
        
        self.decoder_layers = [
            DecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ]
        
        # Output projection
        self.out_proj = np.random.randn(d_model, vocab_size) * np.sqrt(1.0 / d_model)
        self.out_bias = np.zeros(vocab_size)
        
    
    def forward(self, src_tokens, tgt_tokens=None, visualize=False):
        # src-tokens: Input Sequence
        # tgt_tokens: We will use it for training purpose
        # Visualize: Whether we want to visualize the attention weights
        
        # This method will return the next token prediction step
        
        # Get embeddings and we are gonna add the POS - Postional encoding
        src_seq_len = len(src_tokens)
        src_embedded = self.embedding[src_tokens]
        src_pos_enc = self.positional_encoding(src_seq_len, self.d_model)
        src_embedded = src_embedded + src_pos_enc
        
        # Pass through encodder
        # I/P should be passed through each encoder layer
        # Store the Attention weights for visualziation (optional)
        
        enc_output = src_embedded
        attn_maps = []
        
        for i, enc_layer in enumerate(self.encoder_layers): 
            enc_output, attn_weights = enc_layer.forward(enc_output)
            if visualize:
                attn_maps.append((f"Encoder Layer {i+1}", attn_weights))
                
        # If target tokens are provided (for training purpose)
        if tgt_tokens is not None:
            tgt_seq_len = len(tgt_tokens)
            tgt_embedded = self.embedding[tgt_tokens]
            tgt_pos_enc = self.positional_encoding(tgt_seq_len, self.d_model)
            tgt_embedded = tgt_embedded + tgt_pos_enc
            
            # We are creating a casual mask using lower traingular matrix
            tgt_mask = np.tril(np.ones((tgt_seq_len, tgt_seq_len)))
            
            # Pass through decoder
            dec_output = tgt_embedded
            
            for i, dec_layer in enumerate(self.decoder_layers):
                dec_output, self_attn, cross_attn = dec_layer.forward(dec_output, enc_output, tgt_mask=tgt_mask)
                
                if visualize:
                    attn_maps.append((f"Decoder Self Attention Layer {i+1}", self_attn))
                    attn_maps.append((f"Decoder Cross Attention Layer {i+1}", cross_attn))
            
            # Project it to the vocabular representation
            logits = np.dot(dec_output, self.out_proj) + self.out_bias
            
            # Fr visualization
            if visualize:
                self.visualize_attention(attn_maps, src_tokens, tgt_tokens)
                
            return logits
        
        # For inference (without target tokens)
        else:
            return enc_output
        
    
    def positional_encoding(self, seq_len, d_model):
        # return the psotional encoding matrix shape: (seq_len, d_model)
        
        # Initialize positional encoding matrix
        pos_enc = np.zeros((seq_len, d_model))
        
        # Compute the psotional encoding
        # Positions and dividing term
        # Positions: Colum Vector: [0,1,2,....,seq_len-1] represents the pos of each token
        positions = np.arange(seq_len)[:, np.newaxis]
        # We use divind terms for controlling the sine and cosine waves
        div_term = np.exp(np.arange(0, d_model, 2)* - (np.log(10000.0) / d_model))
        # Sin for even indices and Cosine for odd indices
        pos_enc[:, 0::2] = np.sin(positions * div_term)
        pos_enc[:, 1::2] = np.cos(positions * div_term)
        return pos_enc
    
    
    def visualize_attention(self, attn_maps, src_tokens, tgt_tokens=None):
        # Attn_maps: tuple(name, attention_weights)
        # src_tokens and tgt_tokens
        
        num_plots = len(attn_maps)
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
        
        if num_plots == 1:
            axes = [axes]
        
        for i, (name, attn) in enumerate(attn_maps):
            # Take the first head for visualization
            if len(attn.shape) == 4: #multi head attention
                attention = attn[0,0] # First batch, first head
            else:
                attention = attn
                
            # Determine the axis labels based on attention type
            if "Cross-Attn" in name:
                x_labels = [f"{t}" for t in src_tokens]
                y_labels = [f"{t}" for t in tgt_tokens]
            elif "Decoder" in name:
                x_labels = y_labels = [f"{t}" for t in tgt_tokens]
            else: #Encoder
                x_labels = y_labels = [f"{t}" for t in src_tokens]
                                       
            # Plot heatmap
            sns.heatmap(attention, ax=axes[i], cmap="viridis", xticklabels=x_labels, yticklabels=y_labels)
            axes[i].set_title(name)
            axes[i].set_xlabel("Key")
            axes[i].set_ylabel("Query")
        
        plt.tight_layout()
        plt.show()
                                       

    # generate() -> to generate a target sequence using the encoder and decoder architecture
    def generate(self, src_tokens, max_len=1, temperature=0.1):
        # max_len: max number of tokens to generate
        # temperature: control randomness in sampling (lower rate = more confidence/predicatble)
        enc_output = self.forward(src_tokens) #Encode in this step; not decode
        generated = []
        for i in range(max_len):
            # Set decoder input
            curr_seq = generated if generated else[0] #start token
            # Embedding and Position Encoding for the target
            tgt_seq_len = len(curr_seq)
            tgt_embedded = self.embedding[curr_seq]
            tgt_pos_enc = self.positional_encoding(tgt_seq_len, self.d_model)
            tgt_embedded = tgt_embedded + tgt_pos_enc

            # Create a casual mask
            tgt_mask = np.tril(np.ones((tgt_seq_len, tgt_seq_len))) #This actually prevents the decoder from seeing future tokens during generations

            # Pass through decoder layers
            dec_output = tgt_embedded
            for dec_layer in self.decoder_layers:
                # Each layer performs self attention on current sequence
                # Cross attention on ecoder ouput
                dec_output, _, _ = dec_layer.forward(dec_output, enc_output, tgt_mask=tgt_mask)

            logits = np.dot(dec_output[-1], self.out_proj) + self.out_bias # -1 represents last row of the dec_ouput

            # Apply temperature
            logits = logits / temperature # Lower -> More determinstic or controlled outcomes else higher -> More diverse or random

            # Convert to probabilities
            probs = self.softmax(logits)

            # Top K-sampling (for diversity): Instead of sampling over full vocabulary (might be noisy), we sample only from the top 5 likely tokens
            k = 5
            top_indices = ng.argsort(probs)[-k:]
            top_probs = probs[top_indices]
            top_probs = top_probs / np.sum(top_probs)
            next_token = np.random.choice(top_indices, p=top_probs)

            # Append and stop at the end token
            generated.append(next_token)
            if next_token == 1: # we are stopping early if the EOS token is generated
                break

        return generated
    
    
    def softmax(self, x): #X -> vector
        e_x = np.exp(x - np.max(x)) # x - np.max(x) -> sub the max val for preventing ver large exponents, this could cause overflow
        return e_x / e_x.sum() # Normalization -> 0 to 1 instead of input values.
    
    
    def dropout(self, x, training=True): # Regularization technique used during the model training for preventing overfitting cases by randomly setting some inputs to 0
        if not training or self.dropout_prob == 0:
            return x
        
        mask = np.random.binomial(1, 1 - self.dropout_prob, x.shape) / (1 - self.dropout_prob) #Mask value sin binary terms 0 and 1 -> 1: Keep and 0: Drop
        return x * mask
    
    def train_step(self, src_tokens, tgt_tokens, learning_rate=0.001): # Return -> Loss value for the input
        # theta = theta - learning_rate * gradientLoss w.r.t 0
        logits = self.forward(src_tokens, tgt_tokens[:-1]) # We are gonna exclude last target token
        target = tgt_tokens[1:] # Eclcude first token <pad> -> <sos> I Like ML <eos>, tgt_tokens[:-1] -> <sos> I Like ML
        
        loss = 0
        for i, token_logits in enumerate(logits):
            probs = self.softmax(token_logits)
            loss -= np.log(probs[target[i]] + 1e-10) # We are adding small epsilon for numerical stability: Cross ENtropy -> loss = -log(Pmodel(y_true))
        
        return loss 


# In[3]:


class EncoderLayer:
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        #dim_feedforward -> size of the hidden layer in the feed forward network
        # dropout -> preventing the overfitting scenarios
        # Initialize multihead attention
        self.self_attn = MultiHeadAttention(d_model, nhead)
        #Initialize feed forward network (2-layer MLP with ReLu)
        self.ff = FeedForward(d_model, dim_feedforward)
        # Apply the layer normalization before self attention and feedforward (pre-norm)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        # Store the dropout probability
        self.dropout_prob = dropout
        
    # Build the Feedforward layer
    def forward(self, src):
        # Self attention block with pre-morn (More stable training)
        # Src - Input to the encoder layer - usually a sequence matrix of shape (seq_len, d_model)
        src2 = self.norm1.forward(src) # Computing MHA on a normalized input
        attn_output, attn_weights = self.self_attn.forward(src2, src2, src2) # Query, Key and Value
        # Add the residula connection and we are gonna apply dropout
        src = src + self.dropout(attn_output) # O/P attention added back to the input (src) -> TO make the gradient flow is better
            
        # Feed-forward block with pre-norm
        src2 = self.norm2.forward(src) # Normalize the ouput again and we passing it through the feed-forward network
        ff_output = self.ff.forward(src2)
        # We are gonna add another residual connection and dropout layer
        src = src + self.dropout(ff_output) # Residual connection
            
        return src, attn_weights # Returning the op of the encoder layer and the attn_weights for interpretration
        
    def dropout(self, x):
        # Simulate the dropout manually
        if self.dropout_prob == 0:
            return x
            
        mask = np.random.binomial(1, 1 - self.dropout_prob, x.shape) / (1 - self.dropout_prob) # Applying dropout with scaling to keep the expected values constant during training
        return x * mask
            


# In[4]:


# Masked self attention: Attend all the earlier positions only
# Cross Attention - Attend encoder ouputs
# Feed Forward Network: Learn non-linear representations
# Layer Norm + Residual connection + Dropout
class DecoderLayer:
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        # Masked Self Attention (Restricting the decoder to make prediction within the scope)
        self.self_attn = MultiHeadAttention(d_model, nhead)
        # Cross Attention: Decoder queries only from the encoder output
        self.cross_attn = MultiHeadAttention(d_model, nhead)
        # Gonna create a 2-layered feed-forward network
        self.ff = FeedForward(d_model, dim_feedforward)
        # Layer normalization before each block for stability - pre-norm
        #MHA -> layer Norm -> Encodeer + decoder attention -> layer norm -> feed forward -> layer norm -> output (softmax)
        self.norm1= LayerNorm(d_model)
        self.norm2= LayerNorm(d_model)
        self.norm3= LayerNorm(d_model)
        self.dropout_prob = dropout # Storing the dropout probability
        
    # Core logic - Feed Forward Method
    def forward(self, tgt, memory, tgt_mask = None):
        
        # DecoderLayer 1 - Masked MultiHeadAttention
        
        # tgt: Target Token - Target Sequence EMbeddings (Decoder Input)
        # memory: Encoder output (Context from the source sequence)
        # tgt_mask: Casual mask to prevent the model from seeing future tokens
        tgt2 = self.norm1.forward(tgt) # LayerNorm for the input
        self_attn_output, self_attn_weights = self.self_attn.forward(tgt2, tgt2, tgt2, mask=tgt_mask) # Masked Self Attention: Computing attention based on the decoder's past token only
        tgt = tgt + self.dropout(self_attn_output) # Residual: We are adding attention output bck to the orginal and we are applying the dropput for stability
        
        # DecoderLayer 2 - Cross Attention (Decoder attend to encoder): Encoder + Decoer Attention\
        tgt2 = self.norm2.forward(tgt) # LayerNorm on Decoder o/p 
        cross_attn_output, cross_attn_weights = self.cross_attn.forward(tgt2, memory, memory) # Query: tgt2, Key: memory and Value: memory
        tgt = tgt + self.dropout(cross_attn_output)
        
        # DecoderLayer 3 - FeedForward block with pre-norm
        tgt2 = self.norm3.forward(tgt) # Normalizing again before passing to feed forward block
        ff_output = self.ff.forward(tgt2) # Expands and projects back to model dimension
        tgt = tgt + self.dropout(ff_output) # Add residual and dropout layer
        
        return tgt, self_attn_weights, cross_attn_weights # returning final ouput of the decoder layer and also we are returning attention weights
    
    
    def dropout(self, x):
        if self.dropout_prob == 0:
            return x
        
        mask = np.random.binomial(1, 1 - self.dropout_prob, x.shape) / (1 - self.dropout_prob)
        return x * mask


# In[5]:


class MultiHeadAttention:
    def __init__(self, d_model, nhead):
        
        # validate and setup Dimensions
        
        # Each head gets an equal chunk of the model dimension
        assert d_model % nhead == 0, "d_model must be divisble by nhead" # d-model must be divisible by nhead
        #d_k: Dimension of each attention head (512 / 8 = 64)
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        # Initialize Weights - Xavier's or Glorot Initialization (Improving the stability)
        # For normal distribution np.sqrt(2.0 / (d_model + d_model)) -> Var(w) = 2 / (numper of IP + num of Out)
        # 2 / d_model + d_model -> 2 / 2 * d_model = 1 / d_model -> Sqrt (1/d_model)
        scale = np.sqrt(2.0 / (d_model + d_model)) # Preveser consitency and prevent the vanishing gradient pron;e,
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale # Output projection weight
        
    def forward(self, query, key, value, mask=None):
        batch_size = 1
        # Perform dot scale operation for all the params (Q, K, and V)
        Q = np.dot(query, self.W_q) # Q . W_Q
        K = np.dot(key, self.W_k) # K . W_K
        V = np.dot(value, self.W_v) # V . W_v
        
        # Need to reshape for multihead attention
        Q = Q.reshape(batch_size, -1, self.nhead, self.d_k).transpose(0, 2, 1, 3) #d_k -> Dimesion of each attention head
        # T(0,2,1,3) -> Reshaoe and rearrange it so each attention head can work independently
        # Q.transpose(1, 10, 8, 64) -> reshape T(0,2,1,3) -> 0: bacth dimenstion stays first, 2: sequence length , 1: head moves to the second position and 3: dimension per head stays last
        # Result: (1: batch_size,8: head, 10: Sequence len, dimension)
        # (batch_size, nhead, seq_len, d_k)
        K = K.reshape(batch_size, -1, self.nhead, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, -1, self.nhead, self.d_k).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k) # Attention score = Q.KT / Sqrt(d_k)
        # Shape: (batch_size, -1, dimension, nhead)
        
        # Apply mask
        if mask is not None:
            mask = mask.reshape(1,1, *mask.shape)
            scores = scores * mask - 1e9 * (1 - mask) # Large negative value to zero out softmax after masking
            
        # Softmax for scores
        attention_weights = self.softmax(scores) # Converting the scores into probabilities that sum to 1.
        # Calculate weighted sum of values 
        out = np.matmul(attention_weights, V)
        # Shape: batch_size, nhead, seq_len, d_k (We need to project it back to it's orginal representation)

        # Concatenate Heads and we need to project it
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model) # Transposing and rshaping to combine all heads: Shape (seq_len, d_model)
        out = np.dot(out[0], self.W_o) # Final Projection: Concatenating head output
        return out, attention_weights # out -> Final multihead attention op and attention_weights: Interpretation or visuallization
        
    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True)) # Applying softmax along the last dimension with numerical stability
        return e_x / np.sum(e_x, axis=-1, keepdims=True)      
    


# In[6]:


class FeedForward:
    def __init__(self, d_model, dim_feedforward):
        # We are initialize psoition wise feed forward neural network
        # POS - tokens are passed independently through same feed-forward network
        # He Initilization for ReLU # We are 2-layer neural network
        scale1 = np.sqrt(2.0 / d_model)
        scale2 = np.sqrt(2.0 / dim_feedforward) # Keep the variance of the acivatations stable when using it with ReLu activations
        
        # Weights and Bias Initialization - Our 2 layer MLP (Neural Network) with a ReLu in between
        self.linear1 = np.random.randn(d_model, dim_feedforward) * scale1 # 1. First Linear Layer weights (Input -> Hideen Layer)
        self.linear2 = np.random.randn(dim_feedforward, d_model) * scale2 # 2. Second Linear Layer Weights (Hidden Layer -> output)
        self.bias1 = np.zeros(dim_feedforward) # 2. Bias for first linear layer
        self.bias2 = np.zeros(d_model) # 4. Bias for second linear layer
        
    def forward(self, x):
        # Linear transformation
        # Feed Forward Pass: X as an input and return output of the feed forward network
        hidden = np.dot(x, self.linear1) + self.bias1 # Project input into higher dimension space of size (d_model -> dim_feedforward)
        #ReLu activation
        hidden = np.maximum(0, hidden) # Applying non lineairy to add capacity for learning complex features from our IP sequence
        # project back to model original dimension
        out = np.dot(hidden, self.linear2) + self.bias2 # Project back to the original d_model
        return out
    
        # Input (x) (d_model) -> [Linear Layer 1: d_model -> dim_feedforward] -> ReLu -> [Linear Layer 2: dimfeedforward -> d_model] -> output (d_model)


# In[7]:


# LayerNormalization
class LayerNorm:
    def __init__(self, d_model, eps=1e-5):
        # Initialize Layer Normalization (gamma, beta, eps)
        self.gamma = np.ones(d_model) # Scale parameter 
        self.beta = np.zeros(d_model) # Shit parameter
        self.eps = eps
        # Eps -> We are adding a very small number for numerical stability and also for preventing division by zero
        # Gamma and Beta -> Learnable parms that we can scale and shift the normalzied value - allow the network to learn to recover the orignal distribution
        
    def forward(self, x): # takes IP and noralizes each postion/token vector across its features
        # Forward pass thrpugh layer normalization
        # Return vectors
        # Compute Mean and Variance
        mean = np.mean(x, axis=-1, keepdims=True) #axis = -1 when we are computing mean and variance across the feature dimension (d_model) for each token seperately
        var = np.var(x, axis=-1, keepdims=True) # dims=True maintains the shape for better broadcasting
            
        # [[1,2,3], [4,5,6]]: 1x3 | 1x3 -> [[2], [3]]
            
        # Normalize the input: 1. Normalize 2.Scale and 3. Shift
        x_norm = (x - mean) / np.sqrt(var + self.eps) # Subtractying the mean -> center the data
        # Dividing by std using sqrt value btw (var + eps) -> scale to unit variance: If you take sqrt(var) -> std
        # Each token vector has zero mean and unit variance
            
        # Scale and shift
        out = self.gamma * x_norm + self.beta # Applying scaling(gamma) and shit(beta) for each feature dimension
        # Allow the model to learn the optimal normalized distribution.
        return out #Return the layer-normalized vec (same shape as input)


# In[8]:


def run_transformer():
    # Define a larger vocabulary with paragraph-size corpus
    # A key and Pair value
    
    # 36 Tokens
    # Reverse Vocab: reverse mapping
    vocab = {
        "<pad>": 0,
        "<eos>": 1,
        "I": 2,
        "like": 3,
        "to": 4,
        "learn": 5,
        "about": 6,
        "machine": 7,
        "learning": 8,
        "and": 9,
        "artificial": 10,
        "intelligence": 11,
        "because": 12,
        "they": 13,
        "are": 14,
        "fascinating": 15,
        "technologies": 16,
        "of": 17,
        "the": 18,
        "future": 19,
        "transformers": 20,
        "have": 21,
        "revolutionized": 22,
        "natural": 23,
        "language": 24,
        "processing": 25,
        "with": 26,
        "their": 27,
        "attention": 28,
        "mechanism": 29,
        "which": 30,
        "allows": 31,
        "models": 32,
        "understand": 33,
        "context": 34,
        "better": 35
    }
    reverse_vocab = {v: k for k, v in vocab.items()} 
    
    # Initialize the transformer config
    d_model = 64 # Dimension of model -> embeddings and hidden layers
    nhead = 8 # Number of self attention heads
    num_encoder_layers = 3 # Depth of encoder and decoder
    num_decoder_layers = 3
    dim_feedforward = 128 # Size of the feedforward network
    
    transformer = Transformer(
        vocab_size = len(vocab),
        d_model = d_model,
        nhead = nhead,
        num_encoder_layers = num_encoder_layers,
        num_decoder_layers = num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout = 0.1
    )
    
    # Define the input text and tokenize it
    # We are taking the whole inout sentence and we chunkig or tokenizing into work chunks
    input_text = "I like to learn about machine learning and artificial intelligence because they are fascinating technologies"
    input_tokens = [vocab[word] for word in input_text.split()] 
    print("Input Sequence: ")
    print(input_text)
    
    # Expected output sequence
    output_text = "<pad> transformers have revolutionized natural language processing with their attention mechanism"
    output_tokens = [vocab[word] for word in output_text.split()]
    
    print("\n Expected Output")
    print(output_text)
    
    # Training Loop: 
    print("\n Running the training loop")
    learning_rate = 0.01
    epochs = 100
    
    for epoch in tqdm(range(epochs)): # To simply put it it's just a progress bar for iteratbles and loops (Tracking long running tasks)
        loss = transformer.train_step(input_tokens, output_tokens, learning_rate)
        if(epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1} / {epochs}, Loss: {loss: .4f}") 
            #4f -> Retunbr 4 digits after point
                  
    # Implement the forward pass with visualization
    print("\n Running Transformer Forward Pass")
    logits = transformer.forward(input_tokens, output_tokens, visualize=True)
    
    # Print predictons with higher confidence
    print("\n Model Predictions:")
    for i, token_logits in enumerate(logits):
        probs = transformer.softmax(token_logits)
        pred_token = np.argmax(probs) # Order from higher to lower
        print(f"Position {i}: {reverse_vocab[pred_token]} (Probability: {probs[pred_token]:.4f})")
        
    
    # Return both input and output
    return {
        "input": input_text,
        "model": transformer
    }

if __name__ == "__main__":
    results = run_transformer()    

