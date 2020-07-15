'''
Pytorch model train

Author: alex
Created Time: 2020年07月15日 星期三 09时51分39秒
'''
import os
import time
import mlflow
import torch
from torch import nn


class Train:
    cuda = torch.device('cuda')
    cpu = torch.device('cpu')

    def __init__(self, tracking_uri, exp_name, save_path, use_gpu=True):
        """
        :param tracking_uri: str: mlflow.set_tracking_uri, 值如：http://192.168.80.242:20943/
        :param exp_name: str: mlflow实验名称，如果不存在则会创建
        :param save_path: str: 模型保存路径
        """
        mlflow.set_tracking_uri(tracking_uri)
        if mlflow.get_experiment_by_name(exp_name):
            mlflow.set_experiment(exp_name)
        else:
            mlflow.create_experiment(exp_name)

        if use_gpu:
            if torch.cuda.is_available():
                self.device = self.cuda
            else:
                raise Exception('GPU不可用')
        else:
            self.device = self.cpu

        # 配置模型保存路径
        if not os.path.isdir(save_path):
            raise Exception('模型路径不存在：'+save_path)
        self.save_path = save_path

        # 初始化tune log
        self.report = None

    def log_transform(self, transform):
        """记录数据处理与增强参数
        :param transform: str: 数据处理与增强参数
        """
        self.transform = transform

    def set_data(self, trainloader, testloader):
        """设置数据"""
        self.trainloader = trainloader
        self.testloader = testloader

    def set_report_func(self, report):
        """超参指标记录函数"""
        self.report = report

    def run(self, epoch, net, config, optimizer,
            criterion=nn.CrossEntropyLoss()):
        """开始训练"""
        self.config = config
        self.criterion = criterion
        self.optimizer = optimizer
        net_name = net.__class__.__name__
        optim_name = optimizer.__class__.__name__
        # 计算存储路
        file_pre = net_name+'_%s_lr%f' % (optim_name, config['lr'])
        self.last_model = file_pre + '_last.pth'
        self.best_model = file_pre + '_best.pth'
        self.epoch_model = net_name+'_%05d.pth'
        self.last_model = os.path.join(self.save_path, self.last_model)
        self.best_model = os.path.join(self.save_path, self.best_model)
        self.epoch_model = os.path.join(self.save_path, self.epoch_model)
        with mlflow.start_run():
            # 记录训练参数
            mlflow.log_params(config)
            mlflow.log_param('epoch', epoch)
            mlflow.log_param('net', str(net))
            mlflow.log_param('net_name', net_name)
            mlflow.log_param('transform', self.transform)
            mlflow.log_param('criterion', str(criterion))
            mlflow.log_param('optimizer', str(optimizer))
            self.net = net.to(self.device)
            mlflow.log_param('device', self.device)
            if self.device == self.cuda:
                mlflow.log_param('device_name', torch.cuda.get_device_name(0))

            start = time.time()
            self.best_acc = 0
            for epoch_i in range(epoch):
                self.run_epoch(epoch_i)

            mlflow.log_param('run_time', time.time()-start)

        return

    def run_epoch(self, epoch):
        """运行一个epoch"""
        sum_loss = 0.0
        all_loss, all_acc = [], []
        start = time.time()
        for i, data in enumerate(self.trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            loss = self.run_batch(inputs, labels)
            sum_loss += loss
            # 每训练100个batch打印一次平均loss
            if i % 100 == 99:
                avg_loss = sum_loss / 100
                print('[%d, %d] loss: %.03f' % (epoch + 1, i + 1, avg_loss))
                mlflow.log_metric('loss', avg_loss)
                all_loss.append(avg_loss)
                sum_loss = 0.0

        # 每跑完一次epoch测试一下准确率
        with torch.no_grad():
            test_time = time.time()
            train_loss = sum(all_loss)/len(all_loss)
            test_loss, test_acc = self.cal_metric(self.testloader)
            all_acc.append(test_acc)
            mlflow.log_metric('Train_Loss', train_loss, step=epoch)
            mlflow.log_metric('Test_Loss', test_loss, step=epoch)
            mlflow.log_metric('Test_ACC', test_acc, step=epoch)
            print('epoch %d:   Train Loss = %.2f, Test Loss: %.2f, Test Acc = %.2f%%,  Time = %.1fs,  Test Time: %.1fs' % (
                epoch + 1, train_loss, test_loss, 100*test_acc, time.time()-start, time.time()-test_time))

            # 保存模型
            self.save_model(epoch, test_acc)
            if self.report is not None:
                self.report(test_acc)

        return

    def save_model(self, epoch, acc):
        # 最后epoch的模型
        if not os.path.isfile(self.last_model):
            is_first = True
        else:
            is_first = False

        torch.save(self.net.state_dict(), self.last_model)
        if is_first:
            filesize = os.path.getsize(self.last_model)/1024/1024
            mlflow.log_metric('filesize', filesize)       # 记录模型文件大小

        # 保存模型
        if acc > self.best_acc:       # 保存指标最好的模型
            torch.save(self.net.state_dict(), self.best_model)
            self.best_acc = acc
            print('当前最优ACC: %.2f%%, EPOCH: %05d' %
                  (100*self.best_acc, epoch+1))

        # 按epoch保存模型
        torch.save(self.net.state_dict(), self.epoch_model % epoch)

    def run_batch(self, inputs, labels):
        """运行一个批次"""
        # 梯度清零
        self.optimizer.zero_grad()
        # forward + backward
        outputs = self.net(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def cal_metric(self, dataloader):
        correct = 0
        total = 0
        all_loss = []
        for data in dataloader:
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.net(images)
            # 计算loss
            loss = self.criterion(outputs, labels)
            all_loss.append(loss.item())

            # 计算ACC
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        return sum(all_loss) / len(all_loss), int(correct) / total
