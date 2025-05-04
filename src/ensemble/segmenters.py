import torch
from ensemble.ensemble_utils import REGIONS
from uncertainty.test_time_augmentation import tta_variance
from uncertainty.test_time_dropout   import ttd_variance, minmax_uncertainties

class BaseEnsemble:
    def __init__(self, models_dict, perf_weights, device):
        self.models = models_dict
        self.weights = perf_weights  # normalized per class
        self.device  = device

    def fuse(self, batch_image):
        """
        MUST return a tuple (fused_logits, fused_uncertainties)
        fused_uncertainties can be None if not used.
        """
        raise NotImplementedError

    def postprocess(self, fused_logits):
        probs = torch.softmax(fused_logits, dim=0)
        seg   = probs.argmax(dim=0)
        return probs, seg
    
    def prepare_for_eval(self, seg, gt):
        pred_list = [(seg == i).float() for i in range(len(REGIONS))]
        if gt.shape[1] == len(REGIONS):
            gt_list = [gt[:, i].squeeze(0) for i in range(len(REGIONS))]
        else:
            gt_list = [(gt == i).float().squeeze(0) for i in range(len(REGIONS))]
        return pred_list, gt_list

    
class SimpleAverage(BaseEnsemble):
    def fuse(self, image):
        logits_list = []
        for _,inferer in self.models.values():
            logits_list.append(inferer(image).squeeze(0))
        avg_logits = torch.mean(torch.stack(logits_list), dim=0)
        return avg_logits, None
    

class PerfWeighted(BaseEnsemble):
    def fuse(self, image):
        fused = None
        # for each model in the dict
        for name, (_, inferer) in self.models.items():
            # run sliding‐window, squeeze off batch dim → (C,H,W,D)
            logits = inferer(image).squeeze(0)
            # build a (C,1,1,1) weight‐vector from self.weights
            w = torch.tensor(
                [ self.weights[r][name] for r in REGIONS ],
                device=self.device
            ).view(-1,1,1,1)
            weighted = logits * w
            fused = weighted if fused is None else fused + weighted

        return fused, None

class TTAWeighted(BaseEnsemble):
    def __init__(self, models_dict, perf_weights, device, n_iter):
        super().__init__(models_dict, perf_weights, device)
        self.n_iter = n_iter

    def fuse(self, image):
        preds, uncs = {}, {}

        # 1) for each model: run TTA, compute inverse‐variance map
        for name, (_, inferer) in self.models.items():
            m_np, u_np = tta_variance(inferer, image, self.device, n_iterations=self.n_iter)
            m = torch.as_tensor(m_np, device=self.device).squeeze(0)   # (C,H,W,D)
            u = torch.as_tensor(u_np, device=self.device).squeeze(0)   # (C,H,W,D)

            inv = 1.0/(u + 1e-6)
            preds[name] = m * inv
            uncs [name] = u

        # 2) Fuse the “debias‐by‐uncertainty” logits with perf weights
        fused = None
        for name, p in preds.items():
            wlog = torch.stack([
                p[i] * self.weights[REGIONS[i]][name]
                for i in range(len(REGIONS))
            ])
            fused = wlog if fused is None else fused + wlog

        # Temperature optimized by previous temperature scaling
        T_opt = 4.117959976196289
        fused = fused / T_opt

        # 3) Build a fused per‐region uncertainty map
        fused_unc = {}
        for i, region in enumerate(REGIONS[1:], start=1):
            acc = None
            for name in uncs:
                term = uncs[name][i] * self.weights[region][name]
                acc = term if acc is None else acc + term
            fused_unc[region] = minmax_uncertainties(acc.cpu().numpy())

        return fused, fused_unc
    

class TTDWeighted(BaseEnsemble):
    def __init__(self, models_dict, perf_weights, device, n_iter):
        super().__init__(models_dict, perf_weights, device)
        self.n_iter = n_iter

    def fuse(self, image):
        preds, uncs = {}, {}

        # 1) for each model: run TTD, compute inverse‐variance map
        for name, (model, inferer) in self.models.items():
            m_np, u_np = ttd_variance(model, inferer, image, self.device, n_iterations=self.n_iter)
            m = torch.as_tensor(m_np, device=self.device).squeeze(0)   # (C,H,W,D)
            u = torch.as_tensor(u_np, device=self.device).squeeze(0)   # (C,H,W,D)

            inv = 1.0/(u + 1e-6)
            preds[name] = m * inv
            uncs [name] = u

        # 2) Fuse the “debias‐by‐uncertainty” logits with perf weights
        fused = None
        for name, p in preds.items():
            wlog = torch.stack([
                p[i] * self.weights[REGIONS[i]][name]
                for i in range(len(REGIONS))
            ])
            fused = wlog if fused is None else fused + wlog

        # Temperature calibrated exactly as in your script
        T_opt = 4.117968559265137
        fused = fused / T_opt

        # 3) Build a fused per‐region uncertainty map
        fused_unc = {}
        for i, region in enumerate(REGIONS[1:], start=1):
            acc = None
            for name in uncs:
                term = uncs[name][i] * self.weights[region][name]
                acc = term if acc is None else acc + term
            fused_unc[region] = minmax_uncertainties(acc.cpu().numpy())

        return fused, fused_unc

class HybridWeighted(BaseEnsemble):
    def __init__(self, models_dict, perf_weights, device, n_iter):
        super().__init__(models_dict, perf_weights, device)
        self.n_iter = n_iter

    def fuse(self, image):
        preds, uncs = {}, {}

        for name, (model, inferer) in self.models.items():
            # dropout + aug
            m_ttd_np, u_ttd_np = ttd_variance(model, inferer, image, self.device, n_iterations=self.n_iter)
            m_tta_np, u_tta_np = tta_variance(inferer, image, self.device, n_iterations=self.n_iter)

            m_ttd = torch.as_tensor(m_ttd_np, device=self.device).squeeze(0)
            u_ttd = torch.as_tensor(u_ttd_np, device=self.device).squeeze(0)
            u_tta = torch.as_tensor(u_tta_np, device=self.device).squeeze(0)

            # combine the two uncertainty maps
            inv = 1.0/(u_ttd + 1e-6) * 1.0/(u_tta + 1e-6)
            preds[name] = m_ttd * inv
            # average the raw uncertainties for reporting
            uncs[name]  = 0.5*(u_ttd + u_tta)

        fused = None
        for name, p in preds.items():
            wlog = torch.stack([
                p[i] * self.weights[REGIONS[i]][name]
                for i in range(len(REGIONS))
            ])
            fused = wlog if fused is None else fused + wlog

        T_opt = 4.117968559265137
        fused = fused / T_opt

        fused_unc = {}
        for i, region in enumerate(REGIONS[1:], start=1):
            acc = None
            for name in uncs:
                term = uncs[name][i] * self.weights[region][name]
                acc = term if acc is None else acc + term
            fused_unc[region] = minmax_uncertainties(acc.cpu().numpy())

        return fused, fused_unc