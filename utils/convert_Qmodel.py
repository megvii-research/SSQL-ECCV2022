import torch
import torchvision
import os

def convert_Qmodel_to_torch(qmodel, torchname, num_classes=1000):
    # qmodel: quantized model
    # torch model: torchvision model, e.g., resnet50
    ckpt = torch.load(qmodel, map_location="cpu")
    q_dict = ckpt['state_dict']

    for k in list(q_dict.keys()):
        if 'quantizer' in k:
        #if 'quantizer' in k or 'fc' in k:
            del q_dict[k]

    q_keys = list(q_dict.keys())
    print(q_keys, len(q_dict))

    q_index = 0
    torchmodel = torchvision.models.__dict__[torchname](num_classes=num_classes)
    ret_dict = torchmodel.state_dict()

    ret_keys = list(ret_dict.keys())

    #print(ret_dict.keys(), len(ret_dict))
    #for i in range(len(q_keys)):
    #    print(i, q_keys[i], ret_keys[i])
    
    for i, k in enumerate(ret_keys):
        #if 'fc' in k:
            #del ret_dict[k]
        #    continue
        print(i, k, q_keys[i])
        assert ret_dict[k].size()==q_dict[q_keys[i]].size()
        ret_dict[k] = q_dict[q_keys[i]]
    
    ckpt['state_dict'] = ret_dict
    torch.save(ckpt, qmodel.replace('checkpoint', 'torch_checkpoint') )
        
    
if __name__=="__main__":
    #base_root = '/home/caoyunhao/projects/volcano-nn-experiments/lighter-faster-stronger'
    base_root = '/data/'
    #convert_Qmodel_to_torch(os.path.join(base_root, 'train_log/imagenet/RQbyol_randombit_4_16+floatloss_add_float_r50_imagenet_baseline/checkpoint.pth.tar'), 'resnet50')


    #convert_Qmodel_to_torch(os.path.join(base_root, 'train_log/imagenet/RQbyol_randombit_w_2_8_f_4_8+floatloss_add_float_r50_imagenet_baseline/checkpoint.pth.tar'), 'resnet50')
    #convert_Qmodel_to_torch(os.path.join(base_root, 'train_log/imagenet/RQbyol_randombit_w_2_8_f_4_8+floatloss_add_float_r18_imagenet_baseline/checkpoint.pth.tar'), 'resnet18')
    #convert_Qmodel_to_torch(os.path.join(base_root, 'train_log/imagenet/RQbyol_200ep_randombit_w_2_8_f_4_8+floatloss_add_float_r18_imagenet_baseline/checkpoint.pth.tar'), 'resnet18')
    #convert_Qmodel_to_torch(os.path.join(base_root, 'train_log/imagenet/RQbyol_200ep_randombit_w_2_8_f_4_8+floatloss_add_float_r50_imagenet_baseline/checkpoint.pth.tar'), 'resnet50')
    #convert_Qmodel_to_torch(os.path.join(base_root, 'train_log_lars/resnet18_RQbyol_200ep_randombit_w_2_8_f_4_8+floatloss_float_imagenet_linear_eval/checkpoint.pth.tar'), 'resnet18')

    #convert_Qmodel_to_torch(os.path.join(base_root, 'train_log_lars/resnet18_simsiam_rerun_float_imagenet_linear_eval/checkpoint.pth.tar'), 'resnet18')
    
    #convert_Qmodel_to_torch(os.path.join(base_root, 'train_log_lars/resnet50_simsiam_float_imagenet_linear_eval/checkpoint.pth.tar'), 'resnet50')
    #convert_Qmodel_to_torch(os.path.join(base_root, 'train_log_lars/resnet50_RQbyol_200ep_randombit_w_2_8_f_4_8+floatloss_float_imagenet_linear_eval/checkpoint.pth.tar'), 'resnet50')
    convert_Qmodel_to_torch(os.path.join(base_root, 'train_log_lars/resnet50_RQbyol_randombit_4_16+floatloss_float_imagenet_linear_eval/checkpoint.pth.tar'), 'resnet50')


    #convert_Qmodel_to_torch(os.path.join(base_root, 'train_log_finetune/resnet18_RQbyol_randombit_w_2_8_f_4_8+floatloss_float_imagenet_finetune_lr_0.001/checkpoint.pth.tar'), 'resnet18')

    #convert_Qmodel_to_torch(os.path.join(base_root, 'train_log_NEW/resnet18_finetune_cifar10/checkpoint.pth.tar'), 'resnet18', num_classes=10)

    #convert_Qmodel_to_torch(os.path.join(base_root, 'train_log_NEW/resnet18_finetune_cifar10_simsiam/checkpoint.pth.tar'), 'resnet18', num_classes=10)

    #convert_Qmodel_to_torch(os.path.join(base_root, 'train_log_NEW/resnet18_finetune_cifar10_torchvision/checkpoint.pth.tar'), 'resnet18', num_classes=10)

    #convert_Qmodel_to_torch(os.path.join(base_root, 'train_log_NEW/resnet18_finetune_cifar10_torchvision_lr_0.01/checkpoint.pth.tar'), 'resnet18', num_classes=10)