import common.data as data
from tester.CaeReconstructionTester import CaeReconstructionTester


class CaeReconstructionTesterCurve(CaeReconstructionTester):
    def __init__(self, dataloader, path_model, path_outputs_base='/tmp/', normalization_hours_penumbra=10,
                 ta_to_tr_fixed_hours=range(11), ta_to_tr_relative_steps=[0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]):
        CaeReconstructionTester.__init__(self, dataloader, path_model, path_outputs_base=path_outputs_base,
                                         normalization_hours_penumbra=normalization_hours_penumbra)
        self._steps_fixed = ta_to_tr_fixed_hours
        self._steps_relative = ta_to_tr_relative_steps

    def infer_batch(self, batch: dict, step: float):
        dto = self.inference_step(batch, step)
        batch_metrics = self.batch_metrics_step(dto)
        return batch_metrics, dto

    def run_inference(self):
        for batch in self._dataloader:

            # 1) Evaluate on ground truth tA-->tR
            batch_metrics, dto = self.infer_batch(batch, None)
            self.print_inference(batch, batch_metrics, dto)
            self.save_inference(dto, batch)

            # 2) Evaluate metrics curve on fixed tA-->tR: 0 .. 5 hrs
            for step in self._steps_fixed:
                batch_metrics, dto = self.infer_batch(batch, step)
                self.print_inference(batch, batch_metrics, dto, 'ta_to_tr fixed=' + str(step))

            # 3) Evaluate metrics curve on relative tA-->tR:
            ta_to_tr = float(batch[data.KEY_GLOBAL][:, 1, :, :, :])
            for step in self._steps_relative:
                batch_metrics, dto = self.infer_batch(batch, step * ta_to_tr)
                self.print_inference(batch, batch_metrics, dto, 'ta_to_tr ratio=' + str(step) + '\t(' + str(step * ta_to_tr) + ')')

            # 4) Evaluate metrics curve on uniform interval [0,1] between core/penumbra
            to_to_ta = float(batch[data.KEY_GLOBAL][:, 0, :, :, :])
            tr_to_penu = self._normalization_hours_penumbra - to_to_ta
            for step in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                batch_metrics, dto = self.infer_batch(batch, step * tr_to_penu)
                self.print_inference(batch, batch_metrics, dto, 'tr_to_penumbra=' + str(step) + '\t(' + str(step * tr_to_penu) + ')')