import numpy
import medpy.metric.binary as mpm
from common.dto.MetricMeasuresDto import BinaryMeasuresDto


def compute_measures_on_binary_numpy(result, target, threshold=0.5):
    if len(result.shape) == 5:
        result_binary = (result[0, 0, :, :, :] > threshold).astype(numpy.uint8)
    elif len(result.shape) == 3:
        result_binary = (result > threshold).astype(numpy.uint8)
    else:
        raise Exception("Result must be a 3D or 5D tensor")

    if len(target.shape) == 5:
        target_binary = (target[0, 0, :, :, :] > threshold).astype(numpy.uint8)
    elif len(target.shape) == 3:
        target_binary = (target > threshold).astype(numpy.uint8)
    else:
        raise Exception("Target must be a 3D or 5D tensor")

    result = BinaryMeasuresDto(mpm.dc(result_binary, target_binary), numpy.Inf, numpy.Inf)

    if result_binary.any() and target_binary.any():
        result.hd = mpm.hd(result_binary, target_binary)
        result.assd = mpm.assd(result_binary, target_binary)

    return result


# =========================== BATCH LOSS ==========================


from torch.nn.modules.loss import _Loss as LossModule


class BatchDiceLoss(LossModule):
    def __init__(self, label_weights, epsilon=0.0000001, dim=1):
        super(BatchDiceLoss, self).__init__()
        self.epsilon = epsilon
        self._dim = dim
        self._label_weights = label_weights
        print("DICE Loss weights each class' output by", label_weights)

    def forward(self, outputs, targets):
        assert targets.shape[self._dim] == len(self._label_weights), \
            'Ground truth number of labels does not match with label weight vector'
        loss = 0.0
        for label in range(len(self._label_weights)):
            iflat = outputs.narrow(self._dim, label, 1).contiguous().view(-1)
            tflat = targets.narrow(self._dim, label, 1).contiguous().view(-1)
            assert iflat.size() == tflat.size()
            intersection = (iflat * tflat).sum()
            loss += self._label_weights[label] * ((2. * intersection + self.epsilon) /
                                                  ((iflat * iflat).sum() + (tflat * tflat).sum() + self.epsilon))
        return 1.0 - loss