import torch
import torch.nn as nn
from torch.autograd import Function

class BevPoolV2Function(Function):
    @staticmethod
    def bforward(ctx, depth, feat, ranks_depth, ranks_feat, ranks_bev, interval_lengths, interval_starts):
        # Forward pass
        c = feat.size(3)
        n_intervals = interval_lengths.size(0)

        out = torch.zeros((depth.size(0), c, depth.size(2), depth.size(3), feat.size(3)), dtype=torch.float)

        for i in range(n_intervals):
            interval_start = interval_starts[i]
            interval_length = interval_lengths[i]
            for cur_c in range(c):
                for j in range(interval_length):
                    cur_depth = depth[0, ranks_depth[interval_start + j], :, :, :]
                    cur_feat = feat[0, ranks_feat[interval_start + j], :, :, cur_c]
                    out[0, cur_c, :, :, ranks_bev[interval_start]] += cur_depth * cur_feat

        ctx.save_for_backward(depth, feat, ranks_depth, ranks_feat, ranks_bev, interval_lengths, interval_starts)
        ctx.out = out

        return out

    @staticmethod
    def bbackward(ctx, grad_output):
        # Backward pass
        depth, feat, ranks_depth, ranks_feat, ranks_bev, interval_lengths, interval_starts = ctx.saved_tensors
        c, n_intervals = feat.size(3), interval_lengths.size(0)

        depth_grad = torch.zeros_like(depth)
        feat_grad = torch.zeros_like(feat)

        for i in range(n_intervals):
            interval_start = interval_starts[i]
            interval_length = interval_lengths[i]

            for j in range(interval_length):
                cur_rank = ranks_bev[interval_start + j]
                cur_out_grad = grad_output[0, :, :, :, cur_rank]

                for cur_c in range(c):
                    cur_feat_start = feat[0, ranks_feat[interval_start + j], :, :, cur_c]

                    grad_sum = (cur_out_grad[cur_c] * cur_feat_start).sum()
                    depth_grad[0, ranks_depth[interval_start + j], :, :] += grad_sum
                    feat_grad[0, ranks_feat[interval_start + j], :, :, cur_c] += grad_sum

        return depth_grad, feat_grad, None, None, None, None, None



