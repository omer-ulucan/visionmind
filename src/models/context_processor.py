"""
Context Processing Module.
Implements LSTM and attention mechanisms for contextual processing of embeddings.
"""
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

class ContextLSTM(nn.Module):
    """
    Bidirectional LSTM for processing sequential embeddings with attention.
    """
    def __init__(self, config: Dict):
        """
        Initialize the context LSTM.
        
        Args:
            config: Dictionary containing model configuration
                - hidden_size: Size of LSTM hidden states
                - num_layers: Number of LSTM layers
                - dropout: Dropout probability
                - bidirectional: Whether to use bidirectional LSTM
        """
        super(ContextLSTM, self).__init__()
        
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.bidirectional = config['bidirectional']
        
        self.lstm = nn.LSTM(
            input_size=config['hidden_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout'] if config['num_layers'] > 1 else 0,
            bidirectional=config['bidirectional'],
            batch_first=True
        )
        
        # Output dimension is doubled if bidirectional
        self.output_dim = config['hidden_size'] * (2 if config['bidirectional'] else 1)
        
    def forward(
        self, 
        x: torch.Tensor, 
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process sequence of embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            hidden: Optional initial hidden state
            
        Returns:
            Tuple containing:
                - Output tensor of shape (batch_size, seq_len, output_dim)
                - Tuple of final hidden states (h_n, c_n)
        """
        return self.lstm(x, hidden)

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for focusing on relevant information.
    """
    def __init__(self, config: Dict):
        """
        Initialize the multi-head attention module.
        
        Args:
            config: Dictionary containing model configuration
                - num_heads: Number of attention heads
                - head_dim: Dimension of each attention head
                - dropout: Dropout probability
        """
        super(MultiHeadAttention, self).__init__()
        
        self.num_heads = config['num_heads']
        self.head_dim = config['head_dim']
        self.embed_dim = self.num_heads * self.head_dim
        
        self.query = nn.Linear(self.embed_dim, self.embed_dim)
        self.key = nn.Linear(self.embed_dim, self.embed_dim)
        self.value = nn.Linear(self.embed_dim, self.embed_dim)
        
        self.dropout = nn.Dropout(config['dropout'])
        self.output_layer = nn.Linear(self.embed_dim, self.embed_dim)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply multi-head attention.
        
        Args:
            query: Query tensor of shape (batch_size, query_len, embed_dim)
            key: Key tensor of shape (batch_size, key_len, embed_dim)
            value: Value tensor of shape (batch_size, value_len, embed_dim)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, query_len, embed_dim)
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head attention
        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32)
        )
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.embed_dim
        )
        output = self.output_layer(context)
        
        return output

class ContextProcessor(nn.Module):
    """
    Combined LSTM and attention for processing embeddings in context.
    """
    def __init__(self, lstm_config: Dict, attention_config: Dict):
        """
        Initialize the context processor.
        
        Args:
            lstm_config: Configuration for context LSTM
            attention_config: Configuration for multi-head attention
        """
        super(ContextProcessor, self).__init__()
        
        self.lstm = ContextLSTM(lstm_config)
        self.attention = MultiHeadAttention(attention_config)
        
    def forward(
        self,
        embeddings: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process embeddings with context and attention.
        
        Args:
            embeddings: Input embeddings of shape (batch_size, seq_len, hidden_size)
            memory: Optional memory tensor for attention
            mask: Optional attention mask
            
        Returns:
            Tuple containing:
                - Processed embeddings
                - LSTM hidden state for future use
        """
        # Process through LSTM
        lstm_out, (h_n, _) = self.lstm(embeddings)
        
        # Apply attention if memory is provided
        if memory is not None:
            attended = self.attention(lstm_out, memory, memory, mask)
            return attended, h_n
        
        return lstm_out, h_n
