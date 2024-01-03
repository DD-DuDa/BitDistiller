import functools
import inspect

from collections import defaultdict
from typing import Any, Dict, List, Optional, Union
import torch
import random
import numpy as np
# from apex import amp
# from fairscale.nn.data_parallel import (
#     FullyShardedDataParallel as FullyShardedDDP,
#     ShardedDataParallel as ShardedDDP,
# )
# from fairscale.nn.wrap import auto_wrap
from torch import nn
from torch.nn import functional as F, MSELoss
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import Trainer, set_seed
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.trainer_pt_utils import (
    get_module_class_from_name,
)

mse_loss = MSELoss()

class KDTrainer(Trainer):
    def __init__(self, teacher_model, loss_type, mean_prob=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.tlsd = tsld_loss
        self.loss_fct_none = torch.nn.CrossEntropyLoss(reduction="none")
        self.tmp = 1
        self.teacher_model = teacher_model
        # self.reverse_loss = reverse_loss
        self.loss_type = loss_type
        self.mean_prob = mean_prob
        self.ce_loss_none = CrossEntropyLoss(reduction="none")

    def cakld_loss(self, labels, student_logits, teacher_logits, beta_prob):
        mask = (labels != -100)

        # reverse
        teacher_output_log_prob = F.log_softmax(teacher_logits, dim=2)
        # Compute the softmax of the student's logits (approximate distribution)
        student_output_soft = F.softmax(student_logits, dim=2)
        # Calculate the reverse KL Divergence (KL(teacher_logits || student_logits))
        reverse_kl = F.kl_div(teacher_output_log_prob, student_output_soft, reduction="none").sum(-1)

        # forward
        student_output_log_prob = F.log_softmax(student_logits, dim=2)
        teacher_output_soft = F.softmax(teacher_logits, dim=2)
        # Calculate the reverse KL Divergence (KL(teacher_logits || student_logits))
        forward_kl = F.kl_div(student_output_log_prob, teacher_output_soft, reduction="none").sum(-1)

        kl_loss = beta_prob * reverse_kl + (1 - beta_prob) * forward_kl
        kl_loss *= mask
        kl_loss = kl_loss.sum(-1).mean()
        return kl_loss

    def jsd_loss(self, labels, student_logits, teacher_logits, beta_prob):
        mask = (labels != -100)
        student_prob = F.softmax(student_logits, dim=2)
        teacher_prob = F.softmax(teacher_logits, dim=2)

        c_prob = beta_prob * teacher_prob + (1-beta_prob) * student_prob
        c_log_prob = c_prob.log()


        kl_loss_f = beta_prob * F.kl_div(c_log_prob, teacher_prob, reduction="none")
        kl_loss_r = (1 - beta_prob) * F.kl_div(c_log_prob, student_prob, reduction="none")
        kl_loss = kl_loss_f + kl_loss_r

        kl_loss = kl_loss.sum(-1) * mask
        kl_loss = kl_loss.sum(-1).mean()

        return kl_loss

    def ce_loss(self, labels, student_logits, teacher_logits):
        mask = (labels != -100)

        model_output_log_prob = F.log_softmax(student_logits, dim=2)
        real_output_soft = F.softmax(teacher_logits / self.tmp, dim=2)

        # loss = F.kl_div(model_output_log_prob, real_output_soft, reduction="batchmean")
        kl_loss = F.kl_div(model_output_log_prob, real_output_soft, reduction="none")
        kl_loss = kl_loss.sum(-1) * mask
        kl_loss = kl_loss.sum(-1).mean()
        return kl_loss

    def re_loss(self, labels, student_logits, teacher_logits):
        mask = (labels != -100)

        # Compute the log probabilities of the teacher's logits (true distribution)
        teacher_output_log_prob = F.log_softmax(teacher_logits, dim=2)

        # Compute the softmax of the student's logits (approximate distribution)
        student_output_soft = F.softmax(student_logits, dim=2)

        # Calculate the reverse KL Divergence (KL(teacher_logits || student_logits))
        kl_loss = F.kl_div(teacher_output_log_prob, student_output_soft, reduction="none")
        kl_loss = kl_loss.sum(-1) * mask
        kl_loss = kl_loss.sum(-1).mean()
        return kl_loss

    def TLSD_loss(self, labels, student_logits, teacher_logits):
        shift_logits = student_logits[..., :-1, :].contiguous() 
        tc_shift_logits = teacher_logits[..., :-1, :].contiguous() 

        # Step 1. get per-token ce loss with teacher logits
        tc_shift_labels = labels[..., 1:].contiguous().to(labels.device)
        tc_loss_all = self.ce_loss_none(tc_shift_logits.view(-1,tc_shift_logits.size(-1)), tc_shift_labels.view(-1))

        # Step 2. get token-scale with tc_loss_all and temperatured softmax function
        tc_all = tc_loss_all.reshape(tc_shift_logits.shape[0], -1)
        token_scale = torch.nn.functional.softmax(tc_all / 10, dim=-1).clone().detach()

        # Step 3. logit distillation with token-scale
        student_likelihood = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        targets_prob = torch.nn.functional.softmax(tc_shift_logits, dim=-1)
        tsld_loss = (torch.sum((- targets_prob * student_likelihood), dim=-1) * token_scale).sum() # SUM

        return tsld_loss

    def mse_loss(self, student_logits, teacher_logits):
        return mse_loss(student_logits, teacher_logits)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                **inputs
                # **inputs, output_hidden_states=True, output_attentions=True
            )
        teacher_logits = teacher_outputs.get("logits")
        del teacher_outputs

        # forward pass
        student_outputs = model(**inputs)
        # get attributes
        student_logits = student_outputs.get("logits")

        if not return_outputs:
            del student_outputs

        # torch.save(student_logits, "/root/model/acr_duda/code/rej_analysis/sft_student_logits.pt")
        # torch.save(teacher_logits, "/root/model/acr_duda/code/rej_analysis/sft_teacher_logits.pt")
        # torch.save(inputs, "/root/model/acr_duda/code/rej_analysis/sft_inputs.pt")
        # raise 1

        kd_loss = 0.0
        size_average = True
        if model.kd_loss_scale > 0.0:
            if self.loss_type == "reverse":
                kd_loss = self.re_loss(inputs['labels'], student_logits, teacher_logits)
            elif self.loss_type == "forward":
                kd_loss = self.ce_loss(inputs['labels'], student_logits, teacher_logits)
            elif self.loss_type == "tlsd":
                kd_loss = self.TLSD_loss(inputs['labels'], student_logits, teacher_logits)
            elif self.loss_type == "cakld":
                kd_loss = self.cakld_loss(inputs['labels'], student_logits, teacher_logits, self.mean_prob)
            elif self.loss_type == "jsd":
                kd_loss = self.jsd_loss(inputs['labels'], student_logits, teacher_logits, 0.5)
                
        del teacher_logits
        del student_logits

        tok_loss = model.kd_loss_scale * kd_loss
        return (tok_loss, student_outputs) if return_outputs else tok_loss
    
    