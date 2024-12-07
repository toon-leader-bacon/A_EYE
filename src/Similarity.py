
import clip
import torch
import numpy as np


def cosine_similarity(tensor1, tensor2):
    """
    Compute the cosine similarity between two PyTorch tensors.

    Parameters:
    tensor1 (torch.Tensor): First tensor
    tensor2 (torch.Tensor): Second tensor

    Returns:
    torch.Tensor: Cosine similarity between the two tensors, ranging from -1 to 1.
    """
    tensor2 = tensor2.T
    # Compute the dot product
    dot_product = torch.sum(tensor1 * tensor2, dim=-1)

    # Compute the L2 norms
    norm1 = torch.norm(tensor1, dim=-1)
    norm2 = torch.norm(tensor2, dim=-1)

    # Compute the cosine similarity
    cosine_sim = dot_product / (norm1 * norm2)

    return abs(cosine_sim[0])


def euclidean_distance(vec1, vec2):
    """
    Compute the Euclidean distance between two PyTorch tensors.

    Parameters:
    tensor1 (torch.Tensor): First tensor
    tensor2 (torch.Tensor): Second tensor

    Returns:
    float: Euclidean distance between the two tensors
    """
    # Compute the squared differences
    squared_diff = (vec1 - vec2) ** 2

    # Sum the squared differences and take the square root
    distance = torch.sqrt(torch.sum(squared_diff, dim=-1))

    # Return the scalar value
    return distance[0]


def dot_product_similarity(vec1, vec2):
    dot_product = torch.sum(vec1 * vec2, dim=-1)
    return dot_product


class TextSimilarity:
    def __init__(self, text, model, device, similarity_func=dot_product_similarity):
        self.text = text
        self.model = model
        self.device = device
        self.similarity_func = similarity_func

        text_tokens = clip.tokenize([self.text]).to(self.device)
        text_features = self.model.encode_text(text_tokens)
        self.text_features = text_features #/ text_features.norm(dim=-1, keepdim=True)  # l2 norm
        self.similarity = 0

    def __call__(self, vision_model_output):
        # similarity = (vision_model_output @ text_features.T)
        # similarity = cosine_similarity(vision_model_output, text_features.T)
        # similarity = euclidean_distance(vision_model_output, text_features)
        similarity = self.similarity_func(vision_model_output, self.text_features)
        return similarity


class PriorSimilarity:
    def __init__(self, prior_embedding, similarity_func=dot_product_similarity):
        self.prior_embedding = prior_embedding
        self.similarity_func = similarity_func
    
    def __call__(self, vision_model_output):
        similarity = self.similarity_func(vision_model_output, self.prior_embedding)
        return similarity
    

class AveragePriorSimilarity:
    
    def __init__(self, average_embedding, similarity_func=dot_product_similarity):
        self.average_embedding = average_embedding
        # self.average_embedding = torch.mean(torch.Tensor(prior_embeddings), dim=0)
        self.similarity_func = similarity_func
    
    def __call__(self, vision_model_output):
        similarity = self.similarity_func(vision_model_output, self.average_embedding)
        return similarity