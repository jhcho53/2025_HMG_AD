import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import numpy as np

def weighted_sum_loss(predictions, target, weights, criterion):
    """
    각 Task별 Loss를 계산하고 Weighted Sum으로 합산.
    GT 순서가 accel, brake, steer임을 반영.

    Args:
        predictions (torch.Tensor): 모델 예측값 (batch_size, num_tasks)
        target (torch.Tensor): Ground Truth 값 (batch_size, seq_len, num_tasks)
        weights (list or torch.Tensor): 각 Task의 Loss 가중치
        criterion (nn.Module): 손실 함수 (MSELoss 등)

    Returns:
        total_loss (torch.Tensor): Task별 Loss의 Weighted Sum
        task_losses (dict): 각 Task별 Loss 값
    """
    # Ensure weights are a torch.Tensor
    if not isinstance(weights, torch.Tensor):
        weights = torch.tensor(weights, device=predictions.device, dtype=torch.float32)

    # Reduce `target` along the seq_len dimension
    # 평균(pooling) 또는 마지막 프레임 사용
    target_reduced = target.mean(dim=1)  # (batch_size, num_tasks)
    # target_reduced = target[:, -1, :]  # 마지막 프레임만 사용하고 싶다면 이 코드 사용

    # Validate dimensions
    assert predictions.shape == target_reduced.shape, (
        f"Shape mismatch: predictions {predictions.shape}, target_reduced {target_reduced.shape}"
    )
    assert weights.shape[0] == predictions.shape[1], (
        f"Weight shape mismatch: weights {weights.shape}, num_tasks {predictions.shape[1]}"
    )

    # GT 순서: accel, brake, steer
    acceleration_loss = criterion(predictions[:, 0], target_reduced[:, 0])  # accel
    brake_loss = criterion(predictions[:, 1], target_reduced[:, 1])         # brake
    steering_loss = criterion(predictions[:, 2], target_reduced[:, 2])      # steer

    # Weighted Sum of Losses
    total_loss = (
        weights[0] * acceleration_loss +
        weights[1] * brake_loss +
        weights[2] * steering_loss
    )

    # Task별 Loss 반환
    task_losses = {
        "acceleration_loss": acceleration_loss.item(),
        "brake_loss": brake_loss.item(),
        "steering_loss": steering_loss.item()
    }

    return total_loss, task_losses




# Cosine Similarity 기반 Feature Consistency Loss
def feature_consistency_loss(features, labels):
    """
    features: (batch_size, feature_dim)
    labels: (batch_size) -> 같은 label끼리 유사하도록 유도
    """
    # Normalize features
    normalized_features = features / features.norm(dim=1, keepdim=True)

    # Compute pairwise similarity
    similarity_matrix = torch.matmul(normalized_features, normalized_features.T)

    # Create mask for same labels
    labels = labels.unsqueeze(1)
    mask = labels == labels.T

    # Positive pairs (same label)
    positive_pairs = similarity_matrix[mask].mean()

    # Negative pairs (different labels)
    negative_pairs = similarity_matrix[~mask].mean()

    # Loss: maximize positive similarity, minimize negative similarity
    loss = -positive_pairs + negative_pairs
    return loss


# Feature Diversification Loss
def feature_diversification_loss(features):
    """
    features: (batch_size, feature_dim)
    목적: feature들이 서로 다르게 분포하도록 유도
    """
    # Normalize features
    normalized_features = features / features.norm(dim=1, keepdim=True)

    # Compute pairwise cosine similarity
    similarity_matrix = torch.matmul(normalized_features, normalized_features.T)

    # Remove diagonal elements (self-similarity)
    identity = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device)
    diversity_loss = (similarity_matrix - identity).abs().mean()

    return diversity_loss


# Reconstruction Loss for Features
def reconstruction_loss(features, reconstructed_features):
    """
    features: 원본 feature
    reconstructed_features: 복원된 feature
    목적: feature 공간의 정보 보존
    """
    return nn.MSELoss()(features, reconstructed_features)
