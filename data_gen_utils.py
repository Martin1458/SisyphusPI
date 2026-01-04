import random
import torch

# Format: [operand_a, operand_b, equals_token] -> target: result
def to_tensor(n: int, pairs: list[tuple[int, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    to_tensor converts pairs of operands into tensors with format [a, b, equals_token]. (equals_token = n)
    
    :param n: The modulus; N
    :type n: int
    :param pairs: List of tuples containing operand pairs; [(a, b), ...]
    :type pairs: list[tuple[int, int]]
    :return: A tuple containing:
            inputs: Tensor of shape (num_samples, 3) with each row as [a, b, equals_token]
            labels: Tensor of shape (num_samples, 1) with each element as (a + b) % n
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    equals_token: int = n
    inputs: torch.Tensor = torch.tensor([[a, b, equals_token] for a, b in pairs], dtype=torch.long)
    labels: torch.Tensor = torch.tensor([(a + b) % n for a, b in pairs], dtype=torch.long)
    return inputs, labels

def generate_data(n: int, train_pct: float) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """
    generate_data creates all possible addition pairs modulo n, shuffles them,
    and splits them into training and testing datasets based on train_pct.
    
    :param n: The modulus; N
    :type n: int
    :param train_pct: The percentage of data to be used for training
    :type train_pct: float
    :return: Description
    :rtype: tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]
    """
    
    # generate all possible pairs; (a, b) where a, b in [0, n-1]
    all_pairs: list[tuple[int, int]] = [(a, b) for a in range(n) for b in range(n)]
    # shuffle the pairs
    random.shuffle(all_pairs)
    
    # split into train and test sets
    split_idx = int(len(all_pairs) * train_pct)
    train_pairs = all_pairs[:split_idx]
    test_pairs = all_pairs[split_idx:]
    
    return to_tensor(n, train_pairs), to_tensor(n, test_pairs)


if __name__ == "__main__":
    # easy testing
    N: int = 5  # example modulus
    TRAIN_PCT: float = 0.8  # example train percentage
    (train_in, train_lab), (test_in, test_lab) = generate_data(N, TRAIN_PCT)
    print(f"Training samples: {len(train_in)}, Training labels: {len(train_lab)}")
    print(f"Testing samples: {len(test_in)}, Testing labels: {len(test_lab)}")
    print("Sample testing data:")
    for i in range(len(test_in)):
        print(f"Input: {test_in[i].tolist()}, Label: {test_lab[i].item()}")

