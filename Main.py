import torch
import torch.nn as nn
import torch.utils.data.dataloader
import torchvision.transforms as transforms
from Dataset import LevirWhuGzDataset
from tqdm import tqdm
from PIL import Image
from Net import HFANet
import os
import argparse

parser = argparse.ArgumentParser(description='none',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='WHU',
                    help='dataset')
args = parser.parse_args()
DATASET = args.dataset
Net = 'HFANet'

transforms_set = transforms.Compose([
    transforms.ToTensor()
])
transforms_result = transforms.ToPILImage()

train_data = LevirWhuGzDataset(move='train',
                               dataset=DATASET,
                               transform=transforms_set)
test_data = LevirWhuGzDataset(move='test',
                              dataset=DATASET,
                              transform=transforms_set)

BATCH_SIZE = 8
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=BATCH_SIZE // 2,
                                          shuffle=False)

model = HFANet(input_channel=3, input_size=256)
milestone = [10, 15, 30, 40]
optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=0.9, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.1, milestones=milestone)
criterion = nn.BCEWithLogitsLoss()

def confusion_matrix(true_value, output_data, image_size):
    true_positive_sum, true_negative_sum, false_positive_sum, false_negative_sum = 0, 0, 0, 0
    true_value = torch.squeeze(true_value)
    output_data = torch.heaviside(torch.squeeze(output_data), torch.tensor([0], dtype=torch.float32, device='cuda'))
    batch_size = true_value.shape[0]
    for i in range(batch_size):
        union = torch.clamp(true_value[i] + output_data[i], 0, 1)
        intersection = true_value[i] * output_data[i]
        true_positive = int(intersection.sum())
        true_negative = image_size ** 2 - int(union.sum())
        false_positive = int((output_data[i] - intersection).sum())
        false_negative = int((true_value[i] - intersection).sum())
        true_positive_sum += true_positive
        true_negative_sum += true_negative
        false_positive_sum += false_positive
        false_negative_sum += false_negative

    return true_positive_sum, true_negative_sum, false_positive_sum, false_negative_sum


def save_visual_result(output_data, img_sequence):
    output_data = torch.heaviside(torch.squeeze(output_data), torch.tensor([0], dtype=torch.float32, device='cuda'))
    output_data = output_data.cpu().clone()
    batch_size = output_data.shape[0]
    for i in range(batch_size):
        image = transforms_result(output_data[i])
        img_sequence.append(image)
    return img_sequence

def evaluate(tp, tn, fp, fn):
    tp, tn, fp, fn = float(tp), float(tn), float(fp), float(fn)
    oa = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * ((precision * recall) / (precision + recall))
    false_alarm = fp / (tn + fp)
    missing_alarm = fn / (tp + fn)
    CIOU = tp / (tp + fp + fn)
    UCIOU = tn / (tn + fp + fn)
    MIOU = (CIOU + UCIOU) / 2

    return oa, recall, precision, f1, false_alarm, missing_alarm, CIOU, UCIOU, MIOU


def train(train_loader_arg, model_arg, criterion_arg, optimizer_arg, the_epoch, scheduler_arg=None):
    model_arg.cuda()
    model_arg.train()
    with tqdm(total=len(train_loader_arg), desc='Train Epoch #{}'.format(the_epoch + 1)) as t:
        for batch_idx, (img_1, img_2, label) in tqdm(enumerate(train_loader_arg)):
            img_1, img_2, label = img_1.cuda(), img_2.cuda(), label.cuda()
            output = model_arg(img_1, img_2)
            loss = criterion_arg(output, label)
            optimizer_arg.zero_grad()
            loss.backward()
            optimizer_arg.step()
            t.set_postfix({'lr': '%.5f' % optimizer_arg.param_groups[0]['lr'],
                           'loss': '%.4f' % loss.detach().cpu().data})
            t.update(1)
    scheduler_arg.step()


def test(test_loader_arg, model_arg, the_epoch):
    images = []
    images_label = []
    tp, tn, fp, fn = 0, 0, 0, 0
    oa, recall, precision, f1, false_alarm, missing_alarm, ciou, uciou, miou = 0, 0, 0, 0, 0, 0, 0, 0, 0
    model_arg.cuda()
    model_arg.eval()
    with tqdm(total=len(test_loader_arg), desc='Test Epoch #{}'.format(the_epoch + 1)) as t:
        for batch_idx, (img_1, img_2, label) in tqdm(enumerate(test_loader_arg)):
            img_1, img_2, label = img_1.cuda(), img_2.cuda(), label.cuda()
            output = model_arg(img_1, img_2)
            tp_tmp, tn_tmp, fp_tmp, fn_tmp = confusion_matrix(label, output, 256)
            images_label = save_visual_result(label, images_label)
            images = save_visual_result(output, images)
            tp += tp_tmp
            tn += tn_tmp
            fp += fp_tmp
            fn += fn_tmp
            if batch_idx > 200:
                oa, recall, precision, f1, false_alarm, missing_alarm, ciou, uciou, miou = evaluate(tp, tn, fp, fn)
            t.set_postfix({'acc': oa,
                           'f1': '%.4f' % f1,
                           'recall': '%.4f' % recall,
                           'precision': '%.4f' % precision,
                           'false alarm': '%.4f' % false_alarm,
                           'missing alarm': '%.4f' % missing_alarm,
                           'CIOU': '%.4f' % ciou,
                           'UCIOU': '%.4f' % uciou,
                           'MIOU': '%.4f' % miou})
            t.update(1)
    if (the_epoch + 1) >= 1:
        iou_sequence.append(ciou)
        f = open(DATASET + '_' + Net + '.txt', 'a')
        f.write("\"epoch\":\"" + "{}\"\n".format(the_epoch + 1))
        f.write("\"oa\":\"" + "{}\"\n".format(oa))
        f.write("\"f1\":\"" + "{}\"\n".format(f1))
        f.write("\"recall\":\"" + "{}\"\n".format(recall))
        f.write("\"precision\":\"" + "{}\"\n".format(precision))
        f.write("\"false alarm\":\"" + "{}\"\n".format(false_alarm))
        f.write("\"missing alarm\":\"" + "{}\"\n".format(missing_alarm))
        f.write("\"CIOU\":\"" + "{}\"\n".format(ciou))
        f.write("\"UCIOU\":\"" + "{}\"\n".format(uciou))
        f.write("\"MIOU\":\"" + "{}\"\n".format(miou))
        f.close()
        print('max_iou:' + str(max(iou_sequence)) + ' epoch:' + str(iou_sequence.index(max(iou_sequence)) + 1) + '\n')
        if ciou == max(iou_sequence):
            for i in range(len(images)):
                result_label = images_label[i]
                result_image = images[i]
                if not os.path.isdir('vision/' + Net + '/result/' + DATASET):
                    os.makedirs('vision/' + Net + '/result/' + DATASET)
                Image.Image.save(result_image, 'vision/' + Net + '/result/' + DATASET + '/{}.png'.format(i))
                if not os.path.isdir('vision/' + Net + '/label/' + DATASET):
                    os.makedirs('vision/' + Net + '/label/' + DATASET)
                Image.Image.save(result_label, 'vision/' + Net + '/label/' + DATASET + '/{}.png'.format(i))


iou_sequence = []
for epoch in range(100):
    train(train_loader_arg=train_loader,
          model_arg=model,
          criterion_arg=criterion,
          optimizer_arg=optimizer,
          scheduler_arg=scheduler,
          the_epoch=epoch)
    if epoch >= 0:
        test(test_loader_arg=test_loader,
             model_arg=model,
             the_epoch=epoch)
