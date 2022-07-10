import torch
import torch.nn as nn
from torchvision import transforms
from quant_tools.blocks import QuantConv2d, QuantLinear, NAME_QBLOCK_MAPPING
import pulp
import datasets
from importlib import import_module
from typing import Dict
import functools
from utils import logging_information

__all__ = ["HAWQ"]


class HAWQ:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        q_cfg = config.QUANT
        self.max_avg_w_bit = q_cfg.W.BIT
        self.max_avg_a_bit = q_cfg.A.BIT
        mb_cfg = q_cfg.BIT_ASSIGNER
        self.weight_bit_choices = mb_cfg.W_BIT_CHOICES
        self.feature_bit_choices = mb_cfg.A_BIT_CHOICES
        hawq_cfg = mb_cfg.HAWQ
        self.eigen_type = hawq_cfg.EIGEN_TYPE
        self.sensitivity_calc_iter_num = hawq_cfg.SENSITIVITY_CALC_ITER_NUM
        self.add_bit_ascend_limit = hawq_cfg.LIMITATION.BIT_ASCEND_SORT
        self.bit_width_coeff = hawq_cfg.LIMITATION.BIT_WIDTH_COEFF
        self.BOPS_coeff = hawq_cfg.LIMITATION.BOPS_COEFF
        self.eps = 1e-8
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, model_q) -> Dict:
        self.model = model_q
        self.calibration()
        self.get_cache_data()
        loss = self.net_forward_pass()
        self.calc_Qweight_sensitivity(loss)
        self.calc_Qweight_perturbation()
        optimal_bit_conf = self.ILP_search()

        return optimal_bit_conf

    def calibration(self):
        _, eval_preprocess, _ = datasets.build_transforms(self.config)
        if self.config.QUANT.CALIBRATION.TYPE == "tar":
            calibration_dataset = import_module("datasets.tardata").DATASET(
                self.config.QUANT.CALIBRATION.PATH, eval_preprocess
            )
        else:
            raise NotImplementedError(
                "No support {}".format(self.config.QUANT.CALIBRATION.TYPE)
            )
        calibration_dataloader = torch.utils.data.DataLoader(
            calibration_dataset,
            batch_size=self.config.QUANT.CALIBRATION.BATCHSIZE,
            shuffle=False,
            num_workers=self.config.QUANT.CALIBRATION.NUM_WORKERS,
            pin_memory=True,
        )
        self.model.calibration(
            calibration_dataloader, self.config.QUANT.CALIBRATION.SIZE
        )

    def get_cache_data(self):
        # need both data and label for loss calculation, therefore using train dataset
        _, eval_preprocess, _ = datasets.build_transforms(self.config)
        dataset = import_module(self.config.TRAIN.DATASET).DATASET(
            mode="train", transform=eval_preprocess
        )
        # only 1 batch of data used, thus indicate data for HAWQ calibration
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.TRAIN.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.TRAIN.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        # use same data for multiple forward passes in testing
        train_data, train_target = next(iter(data_loader))
        self.train_data = train_data.to(self.device)
        self.train_target = train_target.to(self.device)

    def net_forward_pass(self):
        self.weight_module_attr = {}
        # add hook for Qweight modules to get sorted module names and calc param nums
        def Qmodule_hook(module, data_input, data_output, name):
            if hasattr(module, "weight_quantizer") and module.weight_quantizer:
                module.weight_quantizer.update(module.weight.clone())
                self.weight_module_attr[name] = {
                    "module": module,
                    "param_num": torch.prod(torch.tensor(module.weight.shape))
                    .cpu()
                    .item(),
                }
                h, w = 1, 1  # QLinear
                if isinstance(module, QuantConv2d):
                    h, w = data_output.shape[2:4]
                self.weight_module_attr[name]["GFLOPs"] = (
                    h * w * self.weight_module_attr[name]["param_num"] / 1e9
                )  # G

        Qmodule_hook_handler = []
        for n, m in self.model.named_modules():
            if hasattr(m, "weight_quantizer") and m.weight_quantizer:
                handler = m.register_forward_hook(
                    functools.partial(Qmodule_hook, name=n[6:])
                )
                Qmodule_hook_handler.append(handler)

        criterion = import_module("losses." + self.config.TRAIN.LOSS.NAME).LOSS(
            self.config, self.model
        )
        output = self.model(self.train_data)
        loss = criterion(output, self.train_target)

        for handler in Qmodule_hook_handler:
            handler.remove()

        return loss

    def calc_Qweight_sensitivity(self, loss):
        print("Start Qweight sensitivity calculation!")

        for n, attr in self.weight_module_attr.items():
            w_f = attr["module"].weight
            wf_grads = torch.autograd.grad(loss, w_f, create_graph=True)[0].view(-1)
            sensitivity = self.calc_block_sensitivity(wf_grads, w_f).cpu().item()
            attr["sensitivity"] = sensitivity
            sensitivity_info = n + " sensitivity: {:.4e}".format(attr["sensitivity"])
            print(sensitivity_info)
            logging_information(self.logger, self.config, sensitivity_info)

        if self.add_bit_ascend_limit:
            self.sorted_module_names = [
                k
                for k, _ in sorted(
                    self.weight_module_attr.items(),
                    key=lambda k, v: (v["sensitivity"], k),
                )
            ]

    def rademacher(self, shape, dtype=torch.float32):
        """Sample from Rademacher distribution."""
        rand = ((torch.rand(shape) < 0.5)) * 2 - 1
        return rand.to(dtype).to(self.device)

    def calc_block_sensitivity(self, block_grads, derived_params):
        if self.eigen_type == "max":
            v = torch.randn(block_grads.shape).cuda(self.device, non_blocking=True)
            v = v / torch.norm(v, p=2)
            for _ in range(self.sensitivity_calc_iter_num):
                gv = torch.matmul(block_grads, v)
                gv_grads = torch.autograd.grad(gv, derived_params, retain_graph=True)[
                    0
                ].view(-1)
                gv_grads_norm = torch.norm(gv_grads, p=2)
                v = gv_grads / gv_grads_norm
                max_eigen_value = gv_grads_norm
            return max_eigen_value

        elif self.eigen_type == "avg":
            trace = 0
            v = self.rademacher(block_grads.shape)
            for _ in range(self.sensitivity_calc_iter_num):
                gv = torch.matmul(block_grads, v)
                gv_grads = torch.autograd.grad(gv, derived_params, retain_graph=True)[
                    0
                ].view(-1)
                vHv = torch.matmul(v, gv_grads)
                trace += vHv / self.sensitivity_calc_iter_num
                v = self.rademacher(block_grads.shape)
            avg_eigen_value = trace / gv_grads.shape[0]
            return avg_eigen_value

    def calc_Qweight_perturbation(self):
        for n, attr in self.weight_module_attr.items():
            module = attr["module"]
            bit_perturbations = {}
            for bit in self.weight_bit_choices:
                module.weight_quantizer.set_bit(bit)
                module.weight_quantizer.init_quant_params()
                weight_q = module.weight_quantizer(module.weight)
                delta_weight = weight_q - module.weight
                l2_delta_weight = (delta_weight * delta_weight).sum().cpu().item()
                bit_perturbations[bit] = (
                    l2_delta_weight * self.weight_module_attr[n]["sensitivity"]
                )
            self.weight_module_attr[n]["perturbations"] = bit_perturbations

    def ILP_search(self):
        weight_module_names = list(self.weight_module_attr.keys())
        problem, var = self.set_ILP_target_functions(
            weight_module_names, self.weight_bit_choices, self.weight_module_attr
        )
        problem = self.set_ILP_limitations(
            problem,
            var,
            weight_module_names,
            self.weight_bit_choices,
            self.weight_module_attr,
        )
        best_bit_allocate = self.get_ILP_result(
            problem, weight_module_names, self.weight_bit_choices
        )
        self.logging_search_results(problem)
        return best_bit_allocate

    def set_ILP_target_functions(self, names, bit_choices, attr):
        problem = pulp.LpProblem("bit_allocate", pulp.LpMinimize)
        Qweight_var = pulp.LpVariable.matrix(
            "weight",
            (range(len(names)), range(len(bit_choices))),
            0,
            1,
            pulp.LpInteger,
        )
        target_values = [
            attr[names[i]]["perturbations"][bit_choices[j]] * Qweight_var[i][j]
            for i in range(len(names))
            for j in range(len(bit_choices))
        ]
        problem += pulp.lpSum(target_values)
        return problem, Qweight_var

    def set_ILP_limitations(self, problem, var, names, bit_choices, attr):
        self.total_weight_param_num = sum(
            [attr["param_num"] for _, attr in self.weight_module_attr.items()]
        )
        for i in range(len(names)):  # only 1 bit choice are chosen
            problem += pulp.lpSum(var[i]) == 1
        # add max bit width limitation
        cur_bit_widths = [
            attr[names[i]]["param_num"]
            * bit_choices[j]
            * var[i][j]
            / self.total_weight_param_num
            for i in range(len(names))
            for j in range(len(bit_choices))
        ]
        max_bit_width = self.bit_width_coeff * (self.max_avg_w_bit + self.eps)
        problem += pulp.lpSum(cur_bit_widths) <= max_bit_width
        # add max BOPS limitation
        cur_BOPS = [
            attr[names[i]]["GFLOPs"] * bit_choices[j] * self.max_avg_a_bit * var[i][j]
            for i in range(len(names))
            for j in range(len(bit_choices))
        ]
        max_BOPS = (
            self.BOPS_coeff
            * sum(
                [
                    attr[names[i]]["GFLOPs"] * self.max_avg_w_bit * self.max_avg_a_bit
                    for i in range(len(names))
                ]
            )
            + self.eps
        )
        problem += pulp.lpSum(cur_BOPS) <= max_BOPS

        if self.add_bit_ascend_limit:
            for i in range(len(self.sorted_module_names) - 1):
                problem += pulp.lpDot(
                    bit_choices,
                    var[names.index(self.sorted_module_names[i])],
                ) <= pulp.lpDot(
                    bit_choices,
                    var[names.index(self.sorted_module_names[i + 1])],
                )
        return problem

    def get_ILP_result(self, problem, names, bit_choices):
        problem.solve()
        self.optimal_bit_conf = {}
        for v in problem.variables():
            _, layer_idx, bit_idx = v.name.split("_")
            layer_idx = int(layer_idx)
            bit_idx = int(bit_idx)
            if v.varValue > 0.5:
                self.optimal_bit_conf[names[layer_idx]] = {"w": bit_choices[bit_idx]}

        for n, m in self.model.named_modules():
            name = n[6:]
            if hasattr(m, "output_quantizer") and m.output_quantizer:
                if not name in self.optimal_bit_conf.keys():
                    self.optimal_bit_conf[name] = {"a": self.max_avg_a_bit}
                else:
                    self.optimal_bit_conf[name]["a"] = self.max_avg_a_bit

        print("best bit allocate:")
        logging_information(self.logger, self.config, "best bit allocate:")
        for n in names:
            if (
                "w" in self.optimal_bit_conf[n].keys()
                and "a" in self.optimal_bit_conf[n].keys()
            ):
                info = (
                    n
                    + "  w: "
                    + str(self.optimal_bit_conf[n]["w"])
                    + "  a: "
                    + str(self.optimal_bit_conf[n]["a"])
                )
                print(info)
                logging_information(self.logger, self.config, info)
            elif "w" in self.optimal_bit_conf[n].keys():
                info = n + "  w: " + str(self.optimal_bit_conf[n]["w"])
                print(info)
                logging_information(self.logger, self.config, info)

        return self.optimal_bit_conf

    def logging_search_results(self, problem):
        optimal_avg_w_bit = (
            sum(
                [
                    self.optimal_bit_conf[n]["w"] * attr["param_num"]
                    for n, attr in self.weight_module_attr.items()
                ]
            )
            / self.total_weight_param_num
        )
        optimal_BOPS = sum(
            [
                self.optimal_bit_conf[n]["w"] * self.max_avg_a_bit * attr["GFLOPs"]
                for n, attr in self.weight_module_attr.items()
            ]
        )

        # Print the value of the objective
        logging_information(
            self.logger, self.config, "Status: " + pulp.LpStatus[problem.status]
        )
        logging_information(
            self.logger,
            self.config,
            "Optimal weight perturbation from ILP: "
            + str(pulp.value(problem.objective)),
        )
        logging_information(
            self.logger,
            self.config,
            "Optimal avg weight bit from ILP: " + str(optimal_avg_w_bit),
        )
        logging_information(
            self.logger,
            self.config,
            "optimal total BOPS from ILP: " + str(optimal_BOPS) + " GBOPS",
        )
        uniform_BOPS = sum(
            [
                attr["GFLOPs"] * self.max_avg_w_bit * self.max_avg_a_bit
                for _, attr in self.weight_module_attr.items()
            ]
        )
        logging_information(
            self.logger,
            self.config,
            "uniform {}w{}f BOPS: ".format(self.max_avg_w_bit, self.max_avg_a_bit)
            + str(uniform_BOPS)
            + " GBOPS",
        )
        float_BOPS = sum(
            [attr["GFLOPs"] * 32 * 32 for _, attr in self.weight_module_attr.items()]
        )
        logging_information(
            self.logger, self.config, "float model BOPS: " + str(float_BOPS) + " GBOPS"
        )
