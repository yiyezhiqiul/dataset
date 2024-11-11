import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import csv


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    pr = [int(hist[:, i].sum()) for i in range(hist.shape[1])]
    gt = [int(hist[i, :].sum()) for i in range(hist.shape[0])]
    tp = [int(hist[i, i]) for i in range(hist.shape[0])]
    iu = [tp[i] / (gt[i] + pr[i] - tp[i]) if (gt[i] + pr[i] - tp[i]) != 0 else 0 for i in range(hist.shape[0])]
    return iu


def per_class_PA_Recall(hist):
    pr = [int(hist[:, i].sum()) for i in range(hist.shape[1])]
    gt = [int(hist[i, :].sum()) for i in range(hist.shape[0])]
    tp = [int(hist[i, i]) for i in range(hist.shape[0])]
    recall = [tp[i] / gt[i] if gt[i] != 0 else 0 for i in range(hist.shape[0])]
    return recall


def per_class_Precision(hist):
    pr = [int(hist[:, i].sum()) for i in range(hist.shape[1])]
    gt = [int(hist[i, :].sum()) for i in range(hist.shape[0])]
    tp = [int(hist[i, i]) for i in range(hist.shape[0])]
    precision = [tp[i] / pr[i] if pr[i] != 0 else 0 for i in range(hist.shape[0])]
    return precision


def per_Accuracy(hist):
    return np.trace(hist) / hist.sum()


def compute_mIoU(gt_dir, pred_dir, num_classes, name_classes, label_mapping):
    print('Num classes', num_classes)
    hist = np.zeros((num_classes, num_classes))
    z = 0

    for class_name in os.listdir(gt_dir):
        class_index = label_mapping[class_name]
        for img in os.listdir(os.path.join(gt_dir, class_name)):
            if img.endswith('.png'):
                pred_img = os.path.join(pred_dir, img.replace('_mask.png', '.jpg'))
                gt_img = os.path.join(gt_dir, class_name, img)
                pred = np.array(Image.open(pred_img))
                label = np.array(Image.open(gt_img))
                label = cv2.resize(label, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)

                # 应用标签映射
                label = (label > 0).astype(int) * (class_index + 1)
                pred = np.array([int(x) for x in pred.flatten()])
                hist += fast_hist(label.flatten(), pred, num_classes)

                # 每计算10张就输出一下目前已计算的图片中所有类别平均的mIoU值
                if z % 10 == 0:
                    print(
                        100 * np.nanmean(per_class_iu(hist)),
                        100 * np.nanmean(per_class_PA_Recall(hist)),
                        100 * per_Accuracy(hist)
                    )
                z += 1


    print(z)
    IoUs = per_class_iu(hist)
    PA_Recall = per_class_PA_Recall(hist)
    Precision = per_class_Precision(hist)
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) \
              + '; Recall (equal to the PA)-' + str(round(PA_Recall[ind_class] * 100, 2)) + '; Precision-' + str(
            round(Precision[ind_class] * 100, 2)))

    print('===> mIoU: ' + str(round(np.nanmean(IoUs) * 100, 2)) + '; mPA: ' + str(
        round(np.nanmean(PA_Recall) * 100, 2)) + '; Accuracy: ' + str(round(per_Accuracy(hist) * 100, 2)))
    return np.array(hist, np.int32), IoUs, PA_Recall, Precision


def adjust_axes(r, t, fig, axes):
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])


def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size=12, plt_show=True):
    fig = plt.gcf()
    axes = plt.gca()
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val)
        if val < 1.0:
            str_val = " {0:.2f}".format(val)
        t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
        if i == (len(values) - 1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()


def show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes, tick_font_size=12):
    draw_plot_func(IoUs, name_classes, "mIoU = {0:.2f}%".format(np.nanmean(IoUs) * 100), "Intersection over Union", \
                   os.path.join(miou_out_path, "mIoU.png"), tick_font_size=tick_font_size, plt_show=True)
    print("Save mIoU out to " + os.path.join(miou_out_path, "mIoU.png"))

    draw_plot_func(PA_Recall, name_classes, "mPA = {0:.2f}%".format(np.nanmean(PA_Recall) * 100), "Pixel Accuracy", \
                   os.path.join(miou_out_path, "mPA.png"), tick_font_size=tick_font_size, plt_show=False)
    print("Save mPA out to " + os.path.join(miou_out_path, "mPA.png"))

    draw_plot_func(PA_Recall, name_classes, "mRecall = {0:.2f}%".format(np.nanmean(PA_Recall) * 100), "Recall", \
                   os.path.join(miou_out_path, "Recall.png"), tick_font_size=tick_font_size, plt_show=False)
    print("Save Recall out to " + os.path.join(miou_out_path, "Recall.png"))

    draw_plot_func(Precision, name_classes, "mPrecision = {0:.2f}%".format(np.nanmean(Precision) * 100), "Precision", \
                   os.path.join(miou_out_path, "Precision.png"), tick_font_size=tick_font_size, plt_show=False)
    print("Save Precision out to " + os.path.join(miou_out_path, "Precision.png"))

    with open(os.path.join(miou_out_path, "confusion_matrix.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer_list = []
        writer_list.append([' '] + [str(c) for c in name_classes])
        for i in range(len(hist)):
            writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
        writer.writerows(writer_list)
    print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix.csv"))