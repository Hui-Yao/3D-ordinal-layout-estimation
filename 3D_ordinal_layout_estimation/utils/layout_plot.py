import matplotlib.pyplot as plt
from matplotlib import font_manager
import torch



def layout_plot(history, history_val, metric, metric_val, checkpoint_dir, epoch):
    # myfont = font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc')  # my computer
    myfont = font_manager.FontProperties(fname='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')  # sever
    # plt.figure(figsize=(16,16), dpi=200)
    plt.figure(figsize=(35, 15), dpi=200)

    row = 3
    column = 7

    history_key = list(history.keys())
    for index, single_loss in enumerate(history_key):
        index += 1
        plt.subplot(row, column, index)
        loss_train = history[single_loss]
        loss_val = history_val[single_loss]
        x = range(epoch+1)
        plt.plot(x, loss_train, label = 'train', color = 'r')
        plt.plot(x, loss_val, label = 'val', color = 'g')
        # plt.xlabel('%s'%'epoch', fontproperties=myfont, loc='right')
        # plt.ylabel('%s'%single_loss, fontproperties=myfont, loc='top')
        plt.title('{}/{}'.format(single_loss, 'epoch'), fontproperties=myfont, fontsize=12)
        plt.grid(alpha=0.5)
        plt.legend(prop = myfont,loc = 'upper left', fontsize=12)

    # plt.show()
    # plt.savefig(checkpoint_dir + '/001_loss_epoch_{}'.format(epoch))


    metric_key = list(metric.keys())
    for index, single_metric in enumerate(metric_key):
        start_loc = 8
        if index > 4:
            start_loc = 10

        index += start_loc
        metric_train_ = metric[single_metric]
        metric_val_ = metric_val[single_metric]
        x = range(epoch+1)

        plt.subplot(row, column, index)
        if False:
        # if ('cpa' in single_metric) or ('ciou' in single_metric):
            # print(111, single_metric)
            plt.plot(x, metric_train_, label='train')
            plt.title('{}/{}'.format(single_metric, 'epoch'), fontproperties=myfont, fontsize=12)
            plt.grid(alpha=0.5)
            plt.legend(prop=myfont, loc='upper left', fontsize=12)

            # print('index, loc, before', index, start_loc)
            index += 1
            start_loc += 1
            # print('index, loc, after', index, start_loc)
            plt.subplot(row, column, index)
            plt.plot(x, metric_val_, label='val')

        else:
            plt.plot(x, metric_train_, label='train', color='r')
            plt.plot(x, metric_val_, label='val', color='g')

        plt.title('{}/{}'.format(single_metric, 'epoch'), fontproperties=myfont, fontsize=12)
        plt.grid(alpha=0.5)
        plt.legend(prop = myfont,loc = 'upper left', fontsize=12)


    # plt.show()
    plt.savefig(checkpoint_dir + '/001_epoch_{}'.format(epoch))

