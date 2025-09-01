# No changes needed. This file is correct.
import torch
class Matcher: # ... (code from previous response)
    BELOW_LOW_THRESHOLD, BETWEEN_THRESHOLDS = -1, -2
    def __init__(self, high_threshold: float, low_threshold: float, allow_low_quality_matches: bool = False): self.high_threshold, self.low_threshold, self.allow_low_quality_matches = high_threshold, low_threshold, allow_low_quality_matches
    def __call__(self, match_quality_matrix: torch.Tensor):
        if match_quality_matrix.numel() == 0: return torch.empty((0,), dtype=torch.int64, device=match_quality_matrix.device)
        matched_vals, matches = match_quality_matrix.max(dim=0); all_matches = matches.clone() if self.allow_low_quality_matches else None
        matches[matched_vals < self.low_threshold] = self.BELOW_LOW_THRESHOLD; matches[(matched_vals >= self.low_threshold) & (matched_vals < self.high_threshold)] = self.BETWEEN_THRESHOLDS
        if self.allow_low_quality_matches: self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)
        return matches
    def set_low_quality_matches_(self, matches: torch.Tensor, all_matches: torch.Tensor, match_quality_matrix: torch.Tensor):
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1); gt_pred_pairs_of_highest_quality = torch.where(match_quality_matrix == highest_quality_foreach_gt[:, None]); matches[gt_pred_pairs_of_highest_quality[1]] = all_matches[gt_pred_pairs_of_highest_quality[1]]
class BalancedPositiveNegativeSampler:
    def __init__(self, batch_size_per_image: int, positive_fraction: float): self.batch_size_per_image, self.positive_fraction = batch_size_per_image, positive_fraction
    def __call__(self, matched_idxs):
        pos_idx, neg_idx = [], []
        for matched_idxs_per_image in matched_idxs:
            positive, negative = torch.where(matched_idxs_per_image >= 1)[0], torch.where(matched_idxs_per_image == 0)[0]
            num_pos = min(positive.numel(), int(self.batch_size_per_image * self.positive_fraction)); num_neg = min(negative.numel(), self.batch_size_per_image - num_pos)
            pos_idx.append(positive[torch.randperm(positive.numel(), device=positive.device)[:num_pos]]); neg_idx.append(negative[torch.randperm(negative.numel(), device=negative.device)[:num_neg]])
        return pos_idx, neg_idx