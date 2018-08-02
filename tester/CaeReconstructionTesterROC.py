import common.data as data
from common.model.Cae3D import Cae3D
from tester.CaeReconstructionTester import CaeReconstructionTester


class CaeReconstructionTesterROC(CaeReconstructionTester):
    def __init__(self, dataloader, model: Cae3D, path_model, path_outputs_base='/tmp/', normalization_hours_penumbra=10,
                 ta_to_tr_fixed_hours=range(11), ta_to_tr_relative_steps=[0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]):
        CaeReconstructionTester.__init__(self, dataloader, model, path_model, path_outputs_base=path_outputs_base,
                                         normalization_hours_penumbra=normalization_hours_penumbra)
        self._steps_fixed = ta_to_tr_fixed_hours
        self._steps_relative = ta_to_tr_relative_steps

    def infer_batch(self, batch: dict, step: int):
        dto = self.inference_step(batch, step)
        batch_metrics = self.batch_metrics_step(dto)
        #self.save_inference(dto, batch, suffix='_'+str(step))  # only interested in evaluation metrics
        return batch_metrics, dto

    def run_inference(self):
        for batch in self._dataloader:

            # 1) Evaluate on ground truth tA-->tR
            batch_metrics, dto = self.infer_batch(batch, None)
            self.print_inference(batch, batch_metrics, dto)

            # 2) Evaluate on fixed tA-->tR: 0 .. 5 hrs
            for step in self._steps_fixed:
                batch_metrics, dto = self.infer_batch(batch, step)
                self.print_inference(batch, batch_metrics, dto, 'ta_to_tr fixed=' + str(step))

            # 3) Evaluate on relative tA-->tR:
            ta_to_tr = float(batch[data.KEY_GLOBAL][:, 1, :, :, :])
            for step in self._steps_relative:
                batch_metrics, dto = self.infer_batch(batch, step * ta_to_tr)
                self.print_inference(batch, batch_metrics, dto, 'ta_to_tr ratio=' + str(step) + '\t(' + str(step * ta_to_tr) + ')')