import torch
import torch.nn.functional as F

from utils import *

from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
from loralib import layers as lora_layers

def evaluate_lora(args, clip_model, loader, dataset):
    clip_model.eval()
    with torch.no_grad():
        template = dataset.template[0] 
        texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
        with torch.amp.autocast(device_type="cpu", dtype=torch.float16):
            texts = clip.tokenize(texts).cpu()
            class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.
    tot_samples = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images, target = images.cuda(), target.cuda()
            with torch.amp.autocast(device_type="cpu", dtype=torch.float16):
                image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = image_features @ text_features.t()
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)
    acc /= tot_samples

    return acc


def run_lora(args, clip_model, logit_scale, dataset, test_loader):
    
    VALIDATION = False
    
    # # Textual features
    # print("\nGetting textual features as CLIP's classifier.")
    # textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # # Pre-load test features
    # print("\nLoading visual features and labels from test set.")
    # test_features, test_labels = pre_load_features(clip_model, test_loader)
    
    # test_features = test_features.cuda()
    # test_labels = test_labels.cuda()
 
    # # Zero-shot CLIP
    # clip_logits = logit_scale * test_features @ textual_features
    # zs_acc = cls_acc(clip_logits, test_labels)
    # print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))
    
    # test_features = test_features.cpu()
    # test_labels = test_labels.cpu()
    
    
    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.cpu() 
    
    if args.eval_only:
        load_lora(args, list_lora_layers)
        acc_test = evaluate_lora(args, clip_model, test_loader, dataset)
        print("**** Test accuracy: {:.2f}. ****\n".format(acc_test))
        return
    
    if args.save_path != None:
        save_lora(args, list_lora_layers)
    return
            
    
            
