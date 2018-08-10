import numpy
import medpy.metric.binary as mpm
from common.dto.MetricMeasuresDto import BinaryMeasuresDto
from torch.nn.modules.loss import _Loss as LossModule
from torch.autograd import Variable


class BatchDiceLoss(LossModule):
    def __init__(self, label_weights, epsilon=0.0000001, dim=1):
        super(BatchDiceLoss, self).__init__()
        self._epsilon = epsilon
        self._dim = dim
        self._label_weights = label_weights
        print("DICE Loss weights classes' output by", label_weights)

    def forward(self, outputs, targets):
        assert targets.shape[self._dim] == len(self._label_weights), \
            'Ground truth number of labels does not match with label weight vector'
        loss = 0.0
        for label in range(len(self._label_weights)):
            oflat = outputs.narrow(self._dim, label, 1).contiguous().view(-1)
            tflat = targets.narrow(self._dim, label, 1).contiguous().view(-1)
            assert oflat.size() == tflat.size()
            intersection = (oflat * tflat).sum()
            numerator = 2.*intersection + self._epsilon
            denominator = (oflat * oflat).sum() + (tflat * tflat).sum() + self._epsilon
            loss += self._label_weights[label] * (numerator / denominator)
        return 1.0 - loss


def binary_measures_torch(result, target, cuda, binary_threshold=0.5):
    if cuda:
        result = result.cpu()
        target = target.cpu()

    if isinstance(result, Variable):
        result = result.data
    if isinstance(target, Variable):
        target = target.data

    result = result.numpy()
    target = target.numpy()

    result_binary = (result > binary_threshold).astype(numpy.uint8)
    target_binary = (target > binary_threshold).astype(numpy.uint8)

    result = BinaryMeasuresDto(mpm.dc(result_binary, target_binary),
                               numpy.Inf,
                               numpy.Inf,
                               mpm.precision(result_binary, target_binary),
                               mpm.sensitivity(result_binary, target_binary),
                               mpm.specificity(result_binary, target_binary))

    if result_binary.any() and target_binary.any():
        result.hd = mpm.hd(result_binary, target_binary)
        result.assd = mpm.assd(result_binary, target_binary)

    return result