import torch
import torch.nn.functional as F

def pattern_weaver(n: int, crystal_values: list, dimension: int) -> torch.Tensor:
    """
    Implements a simplified self-attention mechanism for crystal values.
    
    Args:
        n: Number of crystals
        crystal_values: List of crystal values
        dimension: Scaling dimension for attention scores
    
    Returns:
        torch.Tensor of final weighted patterns for each crystal
    """
    # Convert inputs to tensor
    values = torch.as_tensor(crystal_values, dtype=torch.float64)
    # Compute pairwise attention scores and apply softmax
    # Hint: Use torch.outer() for pairwise scores, F.softmax() for normalization,
    #       and torch.matmul() for weighted sum
    output=[]
    print(values.shape)
    for i in range(n):
        pairwise_attention=[torch.dot(values[i].unsqueeze(0),value.unsqueeze(0)) for value in values] 
        normalized_attention= [value/torch.sqrt(torch.tensor(dimension) )for value in pairwise_attention]
        softmax=F.softmax(torch.tensor(normalized_attention),dim=0)
        weighted_sum=torch.matmul(softmax,values)
        output.append(weighted_sum)
    output=torch.stack(output)
    return output
