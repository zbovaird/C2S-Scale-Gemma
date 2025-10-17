# UHG Library API Documentation

## Overview
The UHG (Universal Hyperbolic Geometry) library provides hyperbolic deep learning operations using pure projective geometry. All operations are performed directly in hyperbolic space without tangent space mappings.

## Core Components

### ProjectiveUHG
Main class for projective operations in hyperbolic space.

**Key Methods:**
- `distance(x, y)`: Compute hyperbolic distance between points using Minkowski inner product
- `aggregate(points, weights)`: Aggregate points using weighted projective average
- `normalize(points)`: Normalize points while preserving cross ratios
- `projective_average(points, weights)`: Compute weighted projective average of points
- `cross_ratio(p1, p2, p3, p4)`: Compute cross ratio of four points
- `midpoint(p1, p2)`: Compute midpoint between two points
- `transform(points, matrix)`: Apply projective transformation

### UHGTensor
Extension of PyTorch tensor with UHG operations.

**Key Methods:**
- `cross_ratio()`: Compute cross ratio
- `proj_dist()`: Projective distance
- `proj_transform()`: Projective transformation
- All standard PyTorch tensor operations

### UHG Layers

#### UHGConv
Basic UHG convolution layer.

**Parameters:**
- `in_channels`: Input feature dimension
- `out_channels`: Output feature dimension

**Methods:**
- `forward(x, edge_index)`: Forward pass
- `aggregate_messages(messages, edge_index)`: Aggregate neighbor messages
- `compute_messages(x, edge_index)`: Compute messages between nodes

#### UHGLayerNorm
UHG-aware layer normalization.

**Parameters:**
- `normalized_shape`: Shape of features to normalize
- `eps`: Small value for numerical stability

#### ProjectiveGraphSAGE
Complete GraphSAGE implementation in UHG.

**Parameters:**
- `in_channels`: Input feature dimension
- `hidden_channels`: Hidden layer dimension
- `out_channels`: Output feature dimension
- `num_layers`: Number of layers

**Methods:**
- `forward(x, edge_index)`: Forward pass through all layers

#### ProjectiveSAGEConv
Single GraphSAGE convolution layer.

**Parameters:**
- `input_dim`: Input feature dimension
- `hidden_dim`: Hidden layer dimension
- `output_dim`: Output feature dimension

### UHG Attention

#### UHGMultiHeadAttention
Multi-head attention mechanism in hyperbolic space.

**Parameters:**
- `embed_dim`: Embedding dimension
- `num_heads`: Number of attention heads
- `dropout`: Dropout probability

## Usage Examples

### Basic Distance and Aggregation
```python
from uhg.projective import ProjectiveUHG
import torch

uhg = ProjectiveUHG()

# Create points
points = torch.randn(5, 3)
weights = torch.softmax(torch.randn(5), dim=0)

# Compute distance
dist = uhg.distance(points[0], points[1])

# Aggregate points
agg = uhg.aggregate(points, weights)

# Projective average
avg = uhg.projective_average(points, weights)
```

### Graph Neural Network
```python
from uhg.nn import ProjectiveGraphSAGE
import torch

# Create model
model = ProjectiveGraphSAGE(
    in_channels=64,
    hidden_channels=128,
    out_channels=32,
    num_layers=2
)

# Forward pass
x = torch.randn(100, 64)  # Node features
edge_index = torch.randint(0, 100, (2, 200))  # Edge indices
output = model(x, edge_index)
```

### Single Convolution Layer
```python
from uhg.layers import UHGConv
import torch

# Create layer
conv = UHGConv(in_channels=64, out_channels=32)

# Forward pass
x = torch.randn(100, 64)
edge_index = torch.randint(0, 100, (2, 200))
output = conv(x, edge_index)
```

## Key Features for HGNN Implementation

1. **Distance Computation**: `uhg.distance()` for hyperbolic distances
2. **Message Aggregation**: `uhg.aggregate()` for neighbor aggregation
3. **Layer Normalization**: `UHGLayerNorm` for stable training
4. **Graph Convolution**: `UHGConv` or `ProjectiveSAGEConv` for message passing
5. **Multi-layer Networks**: `ProjectiveGraphSAGE` for complete GNN

## Notes

- All operations preserve hyperbolic geometry properties
- No tangent space mappings required
- Compatible with PyTorch's autograd system
- Supports both CPU and GPU operations
- Cross ratios are preserved under projective transformations
