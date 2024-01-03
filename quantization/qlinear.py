import torch
from torch import Tensor, device, dtype, nn
from quantizer import *

def convertModelToQuant(model, 
                        modules_to_not_convert=["lm_head"], 
                        current_key_name=None, 
                        has_been_replaced=False,
                        compute_dtype=torch.bfloat16, 
                        quant_type="clsq-n2f3", 
                        q_group_size=128):
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)
        if (isinstance(module, nn.Linear)) and name not in modules_to_not_convert:
            in_features = module.in_features
            out_features = module.out_features
            weight = module.weight
            bias = module.bias
            
            model._modules[name] = QLinear(
                in_features,
                out_features,
                module.bias is not None,
                compute_dtype=compute_dtype,
                quant_type=quant_type,
                q_group_size=q_group_size
            )

            model._modules[name].weight = weight
            model._modules[name].bias = bias
            has_been_replaced = True
            # Store the module class in case we need to transpose the weight later
            model._modules[name].source_cls = type(module)
        if len(list(module.children())) > 0:
            _, has_been_replaced = convertModelToQuant(
                module,
                modules_to_not_convert,
                current_key_name,
                has_been_replaced,
                compute_dtype,
                quant_type,
                q_group_size
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced
    
class QLinear(nn.Linear):
    def __init__(self, input_features, output_features, bias=True, compute_dtype=torch.bfloat16, quant_type="ste-n2f3", q_group_size=128, device=None):
        super().__init__(input_features, output_features, bias, device)

        if quant_type == "ste-n2f3":
            self.weight_quantizer = SteN2F3Quantizer(q_group_size=q_group_size)
        elif quant_type == "int2-asym":
            self.weight_quantizer = SteInt2AsymQuantizer(q_group_size=q_group_size)
        else:
            raise ValueError(f"Has no support {quant_type}. Valid quant_type:[ste-n2f3, int2-asym]")
        # self.quant_type = quant_type
        self.compute_dtype = compute_dtype

    def forward(self, x: torch.Tensor):
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        inp_dtype = x.dtype

        if self.compute_dtype is not None:
            x = x.to(self.compute_dtype)

        bias = None if self.bias is None else self.bias.to(self.compute_dtype)
        out = None

        quantize_weight = self.weight_quantizer(self.weight.to(self.compute_dtype))
        out = F.linear(x, quantize_weight, bias).to(inp_dtype)

        return out