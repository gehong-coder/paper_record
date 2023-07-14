# -*- coding:utf-8 -*-
# author:gehong
# datetime:2022/10/6 10:54
# software: PyCharm
import sys
import argparse
import time, os
# 提取模型
import matplotlib.pyplot as plt
import copy
import pickle as pickle
from random import sample
import torchvision
import torch
print(torch.version.cuda)
if (torch.cuda.is_available()):
    print("CUDA 存在")
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import process_data_weibo2 as process_data
from trans_padding import Conv1d as conv1d
from src.early_stopping import *

#多模线性池化
from pytorch_compact_bilinear_pooling import CountSketch, CompactBilinearPooling


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# sys.stdout = Logger("/tmp/pycharm_project_815/src/MBPAM_decoder_weight/log2.txt")  # 保存到D盘
# 可视化
#from tensorboardX import SummaryWriter
#writer = SummaryWriter('runs/multi_fusion')
from sklearn import metrics
from transformers import AutoConfig, TFAutoModel, AutoTokenizer, BertModel, BertTokenizer
#模型位置
# MODEL = '/root/autodl-tmp/pre_trainmodel/ber-base-chinese'
MODEL = 'bert-base-chinese'
N_LABELS = 1
import warnings
warnings.filterwarnings("ignore")


class Rumor_Data(Dataset):
    def __init__(self, dataset):
        self.text = torch.from_numpy(np.array(dataset['post_text']))
        self.image = list(dataset['image'])
        self.mask = torch.from_numpy(np.array(dataset['mask']))
        self.label = torch.from_numpy(np.array(dataset['label']))
        self.event_label = torch.from_numpy(np.array(dataset['event_label']))
        print(
            '数量统计', 'text: %d, image: %d, label: %d, event_label: %d' %
            (len(self.text), len(self.image), len(
                self.label), len(self.event_label)))
        print('TEXT: %d, Image: %d, labe: %d, Event: %d' %
              (len(self.text), len(self.image), len(
                  self.label), len(self.event_label)))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return (self.text[idx], self.image[idx],
                self.mask[idx]), self.label[idx], self.event_label[idx]


# decoder的词典大小
tokenizer = BertTokenizer.from_pretrained(MODEL)
vocab = tokenizer.vocab

params_dict = {
    'latent_dim': 32,
    'combined_fc_out': 64,
    'dec_fc_img_1': 1024,
    'enc_img_dim': 2048,
    'vocab_size': len(vocab),
    'embedding_size': 32,
    'max_len': 20,
    'text_enc_dim': 32,
    'latent_size': 32,
    'hidden_size': 32,
    'num_layers': 1,
    'bidirectional': True,
    'img_fc1_out': 1024,
    'img_fc2_out': 32,
    'fnd_fc1': 64,
    'fnd_fc2': 32
}


#反向传播层--GRL
class ReverseLayerF(Function):
    @staticmethod
    def forward(self, x):
        self.lambd = args.lambd
        return x.view_as(x)

    @staticmethod
    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x):
    return ReverseLayerF.apply(x)


#视觉decoder
class VisualDecoder(nn.Module):
    def __init__(self, latent_dim, dec_fc_img_1, decoded_img):
        super(VisualDecoder, self).__init__()
        self.vis_dec_fc1 = nn.Linear(latent_dim, dec_fc_img_1)
        self.vis_dec_fc2 = nn.Linear(dec_fc_img_1, decoded_img)

    def forward(self, x):
        x = self.vis_dec_fc1(x)
        x = self.vis_dec_fc2(x)
        return x


# 文本decoder
class TextDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, max_len, latent_size,
                 hidden_size, num_layers, bidirectional):
        super(TextDecoder, self).__init__()
        self.max_len = max_len
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.hidden_factor = (2 if bidirectional else 1) * num_layers
        self.text_decoder = nn.LSTM(embedding_size,
                                    hidden_size,
                                    num_layers=num_layers,
                                    bidirectional=bidirectional,
                                    batch_first=True)
        self.latent2hidden = nn.Linear(latent_size,
                                       hidden_size)  ## dec text fc
        self.outputs2vocab = nn.Linear(
            hidden_size * (2 if bidirectional else 1), vocab_size)

    def forward(self, x, max_length):
        hidden = self.latent2hidden(x)
        repeat_hidden = hidden.unsqueeze(1).repeat(
            1, max_length, 1)  ## repeat the hidden input to the max_len
        outputs, _ = self.text_decoder(repeat_hidden)
        outputs = outputs.contiguous()
        b, s, _ = outputs.size()
        logp = nn.functional.log_softmax(self.outputs2vocab(
            outputs.view(-1, outputs.size(2))),
                                         dim=1)
        logp = logp.view(b, s, self.vocab_size)
        return logp


# Neural Network Model (1 hidden layer)
class CNN_Fusion(nn.Module):
    def __init__(self, args):
        super(CNN_Fusion, self).__init__()
        self.args = args
        self.event_num = args.event_num
        vocab_size = args.vocab_size
        emb_dim = args.embed_dim
        C = args.class_num
        self.hidden_size = args.hidden_dim
        self.lstm_size = args.embed_dim
        self.social_size = 19
        convs = []
        self.filter_sizes = [2, 3, 4, 5]
        self.size_pool = 3
        self.drop_rate = 0.2
        self.final_hid = 32
        self.sequence_out = 3840
        # bert
        bert_model = BertModel.from_pretrained(MODEL,
                                               output_hidden_states=True)
        self.bert_hidden_size = args.bert_hidden_dim
        for param in bert_model.parameters():
            param.requires_grad = False
        self.bertModel = bert_model
        '''卷积'''
        self.dropout = nn.Dropout(args.dropout)
        # 4卷积
        self.convs4_2 = nn.Sequential(
            nn.Conv1d(self.sequence_out, 768, 2),  #32 768 191
            nn.BatchNorm1d(768),  ##32 768 191
            nn.LeakyReLU(),  #32 768 191
            nn.MaxPool1d(self.size_pool)  ##32 768 63
        )
        self.convs4_3 = nn.Sequential(nn.Conv1d(self.sequence_out, 768, 3),
                                      nn.BatchNorm1d(768), nn.LeakyReLU(),
                                      nn.MaxPool1d(self.size_pool))
        self.convs4_4 = nn.Sequential(nn.Conv1d(self.sequence_out, 768, 4),
                                      nn.BatchNorm1d(768), nn.LeakyReLU(),
                                      nn.MaxPool1d(self.size_pool))
        self.convs4_5 = nn.Sequential(nn.Conv1d(self.sequence_out, 768, 5),
                                      nn.BatchNorm1d(768), nn.LeakyReLU(),
                                      nn.MaxPool1d(self.size_pool))
        # 2卷积
        self.l2_pool = 768
        self.convs2_1 = nn.Sequential(
            conv1d(self.l2_pool, 768, 3),  #torch.Size([32, 768, 251])
            nn.BatchNorm1d(self.l2_pool),
            nn.LeakyReLU(),
            nn.MaxPool1d(self.size_pool)  #torch.Size([32, 768, 83])
        )
        self.convs2_2 = nn.Sequential(conv1d(self.l2_pool, 768, 3),
                                      nn.BatchNorm1d(self.l2_pool),
                                      nn.LeakyReLU(),
                                      nn.MaxPool1d(self.size_pool))
        # text_append,长度不同
        self.text_flatten = 20736
        self.text_append_layer = nn.Sequential(
            nn.Linear(self.text_flatten, 512), nn.BatchNorm1d(512),
            nn.LeakyReLU(), nn.Dropout(self.drop_rate),
            nn.Linear(512, self.final_hid), nn.BatchNorm1d(32), nn.LeakyReLU())
        # IMAGE
        resnet_1 = torchvision.models.resnet50(pretrained=True)  # 1000
        resnet_1.fc = nn.Linear(2048, 2048)  # 重新定义最后一层
        for param in resnet_1.parameters():
            param.requires_grad = False
        # visual model
        resnet_3 = torchvision.models.resnet50(pretrained=True)
        for param in resnet_3.parameters():
            param.requires_grad = False
        # visual model-分类器取到最后倒数一层
        # resnet_1. =  torch.nn.Sequential(*list(resnet_1.children())[:-1])#提取最后一层了
        self.resnet_1 = resnet_1  # 2048
        # 视觉处理的取到倒数的含有区域的一层
        resnet_3 = torch.nn.Sequential(*list(
            resnet_3.children())[:-2])  # 提取最后一层了
        self.resnet_3 = resnet_3  # 2048*7*7
        # image_append
        self.image_append_layer = nn.Sequential(
            nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Dropout(self.drop_rate), nn.Linear(1024, self.final_hid),
            nn.BatchNorm1d(self.final_hid), nn.LeakyReLU())
        # region_image
        self.region = 49
        self.region_image = nn.Sequential(nn.Linear(2048, self.final_hid),
                                          nn.BatchNorm1d(self.region),
                                          nn.ReLU())
        # attetion att_img
        self.img_dim = 32
        self.att_hid = 32
        self.head = 1
        self.img_key_layer = nn.Linear(self.img_dim,
                                       int(self.att_hid / self.head))
        self.ima_value_layer = nn.Linear(self.img_dim,
                                         int(self.att_hid / self.head))
        self.text_query = nn.Linear(self.final_hid,
                                    int(self.att_hid / self.head))
        # self.score_softmax = nn.Softmax(dim=1)
        # 注意力均值化
        self.att_average = nn.Sequential(nn.Linear(32, self.final_hid),
                                         nn.BatchNorm1d(32), nn.ReLU())
        self.re_Dro = nn.Sequential(nn.Linear(32, 32), nn.BatchNorm1d(32),
                                    nn.LeakyReLU(), nn.Dropout(self.drop_rate))
        # 层激活
        self.layer_norm = nn.LayerNorm(32)

        # attention :attention text
        self.img_query = nn.Linear(self.final_hid,
                                   int(self.att_hid / self.head))
        self.text_key_layer = nn.Linear(self.bert_hidden_size,
                                        int(self.att_hid / self.head))
        self.text_value_layer = nn.Linear(self.bert_hidden_size,
                                          int(self.att_hid / self.head))
        #   soft用上一层的 att_averrage  #均值后用上面的    #层激活
        # self_attention
        self.self_img_query = nn.Linear(32, int(self.att_hid / self.head))
        self.self_img_key = nn.Linear(32, int(self.att_hid / self.head))
        self.self_img_value = nn.Linear(32, int(self.att_hid / self.head))
        # soft同上
        # self_注意力均值化
        self.self_att_average = nn.Sequential(nn.Linear(32, self.final_hid),
                                              nn.BatchNorm1d(49), nn.ReLU())
        # 均值后数句
        self.self_re_Dro = nn.Sequential(nn.Linear(32, 32), nn.BatchNorm1d(49),
                                         nn.LeakyReLU(),
                                         nn.Dropout(self.drop_rate))
        # 层标准化
        self.self_layer_norm = nn.LayerNorm([49, 32])
        # flatten_self
        self.self_faltten = nn.Sequential(nn.Linear(49 * 32, 32),
                                          nn.BatchNorm1d(32), nn.ReLU())
        #文本自注意力机制-text_region
        # self_attention 32 21 768
        self.self_text_query = nn.Linear(768, int(self.att_hid / self.head))
        self.self_text_key = nn.Linear(768, int(self.att_hid / self.head))
        self.self_text_value = nn.Linear(768, int(self.att_hid / self.head))
        # soft同上
        # self_注意力均值化
        self.self_text_att_average = nn.Sequential(
            nn.Linear(32, self.final_hid), nn.BatchNorm1d(27), nn.ReLU())
        # 均值后数句
        self.self_text_re_Dro = nn.Sequential(nn.Linear(32, 32),
                                              nn.BatchNorm1d(27),
                                              nn.LeakyReLU(),
                                              nn.Dropout(self.drop_rate))
        # 层标准化
        self.self_text_layer_norm = nn.LayerNorm([27, 32])
        # flatten_self
        self.self_text_faltten = nn.Sequential(nn.Linear(27 * 32, 32),
                                               nn.BatchNorm1d(32), nn.ReLU())
        #image
        self.multi = nn.Linear(768, 32)
        self.merge = nn.Linear(1024, 32)
        self.mcb = CompactBilinearPooling(27, 49, 32)
        # 融合激活
        self.merge_feature = nn.Sequential(nn.Linear(160, self.final_hid),
                                           nn.Dropout(self.drop_rate))
        # Class Classifier
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1',
                                         nn.Linear(self.hidden_size, 2))
        self.class_classifier.add_module('c_softmax', nn.Softmax(dim=1))

        # Event Classifier
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module(
            'd_fc1', nn.Linear(self.hidden_size, self.hidden_size))
        self.domain_classifier.add_module('d_relu1', nn.LeakyReLU(True))
        self.domain_classifier.add_module(
            'd_fc2', nn.Linear(self.hidden_size, self.event_num))
        #self.domain_classifier.add_module('d_softmax', nn.Softmax(dim=1))

        #decoder结构，使用的是变分自编码器，
        # 需要一个均值和方差来进行得到生成的数据
        self.latent_dim = 32
        self.fnd_fc1 = 64
        self.fnd_fc2 = 32
        self.fc_mu = nn.Linear(160, self.latent_dim)
        self.fc_var = nn.Linear(160, self.latent_dim)
        #激活？
        self.fnd_module = nn.Sequential(
            nn.Linear(self.latent_dim, self.fnd_fc1), nn.Tanh(),
            nn.Linear(self.fnd_fc1, self.fnd_fc2), nn.Tanh(),
            nn.Linear(self.fnd_fc2, 2), nn.Sigmoid())
        #decoder用到上面的结构
        self.text_decoder = TextDecoder(
            params_dict['vocab_size'], params_dict['embedding_size'],
            params_dict['max_len'], params_dict['latent_dim'],
            params_dict['hidden_size'], params_dict['num_layers'],
            params_dict['bidirectional'])
        self.visual_decoder = VisualDecoder(params_dict['latent_dim'],
                                            params_dict['dec_fc_img_1'],
                                            params_dict['enc_img_dim'])

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (to_var(torch.zeros(1, batch_size, self.lstm_size)),
                to_var(torch.zeros(1, batch_size, self.lstm_size)))

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (sample number,hidden_dim, length)
        # x = F.avg_pool1d(x, x.size(2)).squeeze(2)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)

        return x

    # gh:decoder
    def decode(self, z, max_len):
        recon_text = self.text_decoder(z, max_len)

        recon_img = self.visual_decoder(z)

        return [recon_text, recon_img]

    def forward(
        self, text, image, mask
    ):  # torch.Size([32, 192]) torch.Size([32, 3, 224, 224]) torch.Size([32, 192])
        """GH"""
        # 改进点1：基于双分支对抗网络的虚假新闻检测
        text_decoder = text
        # print(text.size())# 32
        # print(image.size())
        out = self.bertModel(text)  # berts输出
        last_hidden_state = out[0]
        '''bert隐藏层四层'''
        all_hidden_state = out[2]  # bert模型的原因，因为，out【2】是所有层的输出
        '''将bert的后面四层堆叠'''
        sequence_output = torch.cat(
            (all_hidden_state[-1], all_hidden_state[-2], all_hidden_state[-3],
             all_hidden_state[-4], all_hidden_state[-5]), 2)
        #print(sequence_output.shape)    #torch.Size([32, 192, 3840])
        # 由于torch卷积的原因，需要将max_len和词向量维度进行转变使用permute函数
        sequence_output = sequence_output.permute(
            0, 2, 1)  #torch.Size([32, 3840, 192])
        #print(sequence_output.shape)
        '''四卷积'''
        convs = []
        l_pool_2 = self.convs4_2(sequence_output)  #torch.Size([32, 768, 63])
        #print(l_pool_2.shape)
        convs.append(l_pool_2)
        l_pool_3 = self.convs4_3(sequence_output)  #torch.Size([32, 768, 63])
        #print(l_pool_3.shape)
        convs.append(l_pool_3)
        l_pool_4 = self.convs4_4(sequence_output)  #torch.Size([32, 768, 63])
        #print(l_pool_4.shape)
        convs.append(l_pool_4)
        l_pool_5 = self.convs4_5(sequence_output)  #torch.Size([32, 768, 62])
        #print(l_pool_5.shape)
        convs.append(l_pool_5)
        '''拼接4卷积-convs'''
        l2_pool = torch.cat(convs, dim=2)
        #print(l2_pool.shape)    #torch.Size([32, 768, 251])
        '''两卷积'''
        # 卷积 - 批标准化 - 激活- 池化
        l2_pool = self.convs2_1(l2_pool)  #torch.Size([32, 768, 83])
        #print(l2_pool.shape)
        l2_pool = self.convs2_2(l2_pool)  #torch.Size([32, 768, 27])

        #print(l2_pool.shape)
        '''卷积完成，将词向量的维度和max——len进行调换位置-TM：【batch，区域数，向量维度】'''
        Tm = l2_pool.permute(0, 2, 1)  #torch.Size([32, 27, 768])
        #print(Tm.shape)batch,chanel,dim
        '''展平,（batch 不展平）此时后面的两个维度交不交换没有必要性，因为后面要进行flatten操作'''
        text = torch.flatten(Tm, start_dim=1,
                             end_dim=2)  #torch.Size([32, 20736])
        #print(text.shape)
        '''text双流1:tm'''
        #multi_text = self.multi(text)   #torch.Size([32, 1568])
        multi_text = self.multi(Tm)  #32 27 768--32 27 32
        #print(multi_text.shape)
        multi_text = multi_text.permute(0, 2, 1)  #batch,dim,chanel
        #print(multi_text.shape)
        merge = []
        '''text双流2:text_append[batch_size,32],直接拼接'''
        text_append = self.text_append_layer(text)
        #print(text_append.shape)    #torch.Size([32, 32])
        merge.append(text_append)
        '''image_双流1：image_append——倒数一层vgg'''
        image_1 = self.resnet_1(image)  #   torch.Size([32, 2048])
        cnn_enc_img = image_1
        #print(image_1.shape)
        image_append = self.image_append_layer(image_1)  #torch.Size([32, 32])
        #print(image_append.shape)
        merge.append(image_append)
        '''image双柳2：区域image——倒数3层'''
        image_3 = self.resnet_3(image)  # torch.Size([32, 2048, 7, 7])
        #print(image_3.shape)
        image_3 = image_3.permute(0, 2, 3, 1)  # torch.Size([32, 7, 7, 2048])
        image_3 = torch.reshape(image_3,
                                ((image_3.shape[0], -1, image_3.shape[-1])))
        #print(image_3.shape)#torch.Size([32, 49, 2048])
        image = self.region_image(image_3)  #torch.Size([32, 49, 32])
        #print(image.shape)
        Im = image
        #multi_image = torch.flatten(Im, start_dim=1, end_dim=2)  #torch.Size([32, 1568])
        multi_image = image.permute(0, 2, 1)  #32 32,49
        #print(multi_image.shape)

        # 改进点2： 基于组合式融合机制的虚假新闻检测
        '''模态间：多模线性池化块'''
        # print("-" * 50, "multi-model", "-" * 50)
        x = multi_text
        #print(x.shape)  # torch.Size([32, 32, 27])
        y = multi_image
        #print(y.shape)  # torch.Size([32, 32, 49])
        Merge_feture = self.mcb(x, y)
        #print(Merge_feture.shape)  # torch.Size([32, 32, 32])
        Merge_feture = torch.flatten(Merge_feture, start_dim=1, end_dim=2)
        Merge_feture = self.merge(Merge_feture)  #
        #print(Merge_feture.shape)
        merge.append(Merge_feture)
        head = 1
        att_layers = 1
        att_hid = 32
        '''模态内：自注意力机制-Im-Im'''
        in_AttSelf_key = Im  # torch.Size([32, 49, 32])
        in_AttSelf_query = Im  # torch.Size([32, 49, 32])
        for layer in range(att_layers):
            self_att_img = []
            for _ in range(head):
                self_img_query = self.self_img_query(in_AttSelf_query)
                self_img_key = self.self_img_key(in_AttSelf_key)
                self_img_value = self.self_img_value(in_AttSelf_key)
                # torch.Size([32, 49, 32])  torch.Size([32, 49, 32])
                '''计算sccore Q*K-公式第一步'''
                score = torch.tensordot(self_img_query,
                                        self_img_key,
                                        dims=([2],
                                              [2]))  # torch.Size([32, 32, 21])
                # print(score.shape)  # torch.Size([32, 49,32, 49])
                '''改变合并方法'''
                score = torch.stack(
                    [score[i, :, i, :] for i in range(len(score))])
                # print(score.shape)  # torch.Size([32, 49,49])
                # print("score.shape")
                '''公式第二步,score/根号下d 维度-维度不匹配的广播机制'''
                # 如果是y一个数，x所有元素除以y
                score = torch.div(score, np.sqrt(att_hid / head))  # [32,49,49]
                score = F.leaky_relu(score)
                '''公式第三步-score * v'''
                # score：torch.Size([32, 49,49]) image_value = torch.Size([32, 49, 32])
                attention = torch.tensordot(score,
                                            self_img_value,
                                            dims=([2], [1]))
                # print(attention.shape)  # torch.Size([32, 49, 32,32])
                attention = torch.stack(
                    [attention[i, :, i, :] for i in range(len(attention))])
                # print(attention.shape)  #   torch.Size([32, 49, 32])
                '''得出询问后的自己后的 att'''
                self_att_img = attention
                '''将注意力均值化-无法平均-也就是多个列表，对应的元素相加求平均在进行成为当前元素'''
                self_att_img21 = self.self_att_average(self_att_img)
                self_att_img22 = self.self_att_average(self_att_img)
                self_att_img23 = self.self_att_average(self_att_img)
                self_att_img24 = self.self_att_average(self_att_img)
                # 均值
                self_att_img2 = self_att_img21.add(self_att_img22).add(
                    self_att_img23).add(self_att_img24)
                self_att_img2 = torch.div(self_att_img2, 4)
                '''均值后数据'''
                self_att_img2 = self.self_re_Dro(self_att_img2)
                '''将注意力后的数据相加:image_+self_att_img'''
                self_att_img = torch.add(in_AttSelf_query, self_att_img2)
                # print(self_att_img.shape) # [32,49,32]
                '''层标准化'''
                self_att_img = self.self_layer_norm(self_att_img)
                '''不能将他直接merge进去因为，维度不同，faltten之后进行输入'''
                # print(self_att_img.shape)# [32,49,32]
                in_AttSelf_query = self_att_img
                inp_AttSelf_key = self_att_img
        '''将后面的两维度flatten，变成二维'''
        self_att_img = torch.flatten(self_att_img, start_dim=1,
                                     end_dim=2)  # end_dim=-1
        self_att_img = self.self_faltten(self_att_img)
        '''此时向量的维度是正常的'''
        # print(self_att_img.shape)#[32,32]
        merge.append(self_att_img)
        '''text自注意力机制'''
        in_Self_TEXT_key = Tm  #torch.Size([32, 27, 768])
        #print(in_Self_TEXT_key.shape)
        in_Self_TEXT_query = Tm  #
        for layer in range(att_layers):
            self_text_att_img = []
            for _ in range(head):
                self_text_query = self.self_text_query(
                    in_Self_TEXT_query)  ##torch.Size([32, 27, 32])
                self_text_key = self.self_text_key(
                    in_Self_TEXT_key)  #torch.Size([32, 27, 32])
                self_text_value = self.self_text_value(
                    in_Self_TEXT_key)  #torch.Size([32, 27, 32])
                '''计算sccore Q*K-公式第一步'''
                score = torch.tensordot(self_text_query,
                                        self_text_key,
                                        dims=([2],
                                              [2]))  # torch.Size([32, 32, 21])
                #print(score.shape)  #torch.Size([32, 27, 32, 27])
                '''改变合并方法'''
                score = torch.stack(
                    [score[i, :, i, :] for i in range(len(score))])
                #print(score.shape)  #torch.Size([32, 27, 27])
                '''公式第二步,score/根号下d 维度-维度不匹配的广播机制'''
                # 如果是y一个数，x所有元素除以y
                score = torch.div(score, np.sqrt(att_hid / head))
                score = F.leaky_relu(score)
                '''公式第三步-score * v'''
                attention = torch.tensordot(score,
                                            self_text_value,
                                            dims=([2], [1]))
                attention = torch.stack(
                    [attention[i, :, i, :] for i in range(len(attention))])
                '''得出询问后的自己后的 att'''
                self_text_att_img = attention
                '''将注意力均值化-无法平均-也就是多个列表，对应的元素相加求平均在进行成为当前元素'''
                self_text_att21 = self.self_text_att_average(self_text_att_img)
                self_text_att22 = self.self_text_att_average(self_text_att_img)
                self_text_att23 = self.self_text_att_average(self_text_att_img)
                self_text_att24 = self.self_text_att_average(self_text_att_img)
                # 均值
                self_text_att2 = self_text_att21.add(self_text_att22).add(
                    self_text_att23).add(self_text_att24)
                self_text_att2 = torch.div(self_text_att2, 4)
                '''均值后数据'''
                self_text_att2 = self.self_text_re_Dro(self_text_att2)
                '''将注意力后的数据相加:image_+self_att_img'''
                self_text_att = torch.add(self_text_query,
                                          self_text_att2)  # [32,27,32]
                '''层标准化'''
                self_text_att = self.self_text_layer_norm(self_text_att)
                '''不能将他直接merge进去因为，维度不同，faltten之后进行输入'''
                in_Self_TEXT_query = self_text_att
                in_Self_TEXT_key = self_text_att
        '''将后面的两维度flatten，变成二维'''
        self_text_att = torch.flatten(self_text_att, start_dim=1,
                                      end_dim=2)  # end_dim=-1
        self_text_att = self.self_text_faltten(self_text_att)
        '''此时向量的维度是正常的'''
        #print(self_text_att.shape)  #torch.Size([32, 32])
        merge.append(self_text_att)
        #print("-" * 50, "结束融合", "-" * 50)
        '''1。共有5部分数据- 32*32 ,32-160'''
        feature_merge = torch.cat(merge, dim=1)  # 32, 160

        # 改进点3：基于基于变分自编码器进行多任务学习
        """2。decoder结构，VAR"""
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(feature_merge)  # 32 32
        #print(mu.shape) # 32 32
        log_var = self.fc_var(feature_merge)  # 32 32
        #print(log_var.shape)
        z = self.reparameterize(mu, log_var)  #32 32
        #print(z.shape)
        #重建的text和图像
        recon_text, recon_img = self.decode(z, text_decoder.shape[1])
        # print(recon_text.shape)#torch.Size([32, 192, 21128]), 因为重建的文本，每个都是单词
        # print(recon_img.shape)# torch.Size([32, 4096])
        """3。虚假新闻分类器"""
        fnd_out = self.fnd_module(z)  #torch.Size([32, 2])
        # fnd_out = self.merge_feature(feature_merge)#torch.Size([32, 2])
        # print(fnd_out.shape)#
        feature_merge = self.merge_feature(feature_merge)
        fnd_out = self.class_classifier(feature_merge)
        class_output = fnd_out
        """4。域分类器"""
        reverse_feature = grad_reverse(feature_merge)
        domain_output = self.domain_classifier(reverse_feature)
        #print(cnn_enc_img.shape)#torch.Size([32, 2048])

        return [
            fnd_out, recon_text, recon_img, mu, log_var, cnn_enc_img,
            class_output, domain_output
        ]


"""        
text:最原始的text
fnd_out, 分类输出，虚假 32 2
recon_text, 重建的文本， 32 192 21128
recon_img,  重建的图像， 32 2048
mu, 变分自编码器的的均值
log_var, 变分自编码器的方差
cnn_enc_img, 原始的图像 32 2048
class_output, 
domain_output
注意：
recon_text vs text
recon_img vs cnn_enc_img
"""


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def to_np(x):
    return x.data.cpu().numpy()


def select(train, selec_indices):
    temp = []
    for i in range(len(train)):
        print("length is " + str(len(train[i])))
        print(i)
        ele = list(train[i])
        temp.append([ele[i] for i in selec_indices])
    return temp


def make_weights_for_balanced_classes(event, nclasses=15):
    count = [0] * nclasses
    for item in event:
        count[item] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(event)
    for idx, val in enumerate(event):
        weight[idx] = weight_per_class[val]
    return weight


def split_train_validation(train, percent):
    whole_len = len(train[0])

    train_indices = (sample(range(whole_len), int(whole_len * percent)))
    train_data = select(train, train_indices)
    print("train data size is " + str(len(train[3])))

    validation = select(train, np.delete(range(len(train[0])), train_indices))
    print("validation size is " + str(len(validation[3])))
    print("train and validation data set has been splited")

    return train_data, validation


#损失函数
fnd_loss_fn = nn.CrossEntropyLoss()
recon_text_loss = nn.NLLLoss()


def loss_function(ip_text, ip_img, ip_label, mu, log_var, rec_text, rec_img,
                  fnd_label, lambda_wts):
    """
    Computes the VAE loss function.
    KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
    """
    # print("loss")
    # print(fnd_label.shape) torch.Size([32, 2])
    # print(ip_label.shape) torch.Size([32])
    # print(ip_img.shape)torch.Size([32, 2048])
    # print(rec_img.shape)torch.Size([32, 2048])
    # print(rec_text.shape)torch.Size([32, 192, 21128])
    # print(ip_text.shape)torch.Size([32, 192])
    fnd_loss = fnd_loss_fn(fnd_label, ip_label)  # 32 2, 32 1
    recons_loss = F.mse_loss(ip_img, rec_img)  # 32 2048, 32 2048
    rec_text = rec_text.view(-1,
                             rec_text.size(2))  # 32 192 21128-->32 * 192 21128
    #print(rec_text.shape)torch.Size([6144, 21128])
    ip_text = ip_text.view(-1)  # 32 192--> 32*192
    #print(ip_text.shape)torch.Size([6144])
    text_loss = recon_text_loss(rec_text, ip_text)  # 32 * 192 21128 , 32*192
    kld_loss = torch.mean(
        -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)
    loss = lambda_wts['fnd'] * fnd_loss + lambda_wts[
        'img'] * recons_loss + lambda_wts['kld'] * kld_loss + lambda_wts[
            'text'] * text_loss
    #print(fnd_loss,recons_loss,text_loss, kld_loss)
    return fnd_loss, recons_loss, text_loss, kld_loss, loss


# fnd_loss 分类损失，recons_loss重建 图像损失， text_loss 重建文本损失 kld_loss 分布差异


def main(args):
    print("-" * 50, "开始载入数据", "-" * 50)
    print('loading data')
    train, validation, test = load_data(args)

    test_id = test['post_id']
    '''通过Rumor——data进行找到数据存在的所有项'''
    train_dataset = Rumor_Data(train)

    validate_dataset = Rumor_Data(validation)

    test_dataset = Rumor_Data(test)

    # Data Loader (Input Pipeline)，构造数据，定义data——set和batch
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)

    validate_loader = DataLoader(dataset=validate_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False)
    # test也是按照batch送入的
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False)
    print("-" * 50, "开始生成模型", "-" * 50)
    print('building model')
    model = CNN_Fusion(args)

    print("-" * 50, "打印模型", "-" * 50)
    print("-" * 50, "模型结构", "-" * 50)
    print(model)
    if torch.cuda.is_available():
        print("CUDA ok")
        model.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        list(model.parameters())),
                                 lr=args.learning_rate)
    print("bach总数：", "训练集" + str(len(train_loader)),
          "验证集" + str(len(validate_loader)), "测试集" + str(len(test_loader)))
    iter_per_epoch = len(train_loader)
    print("-" * 50, "开始训练", "-" * 50)
    # Start training loop
    lambda_wts = {'fnd': 1, 'img': 10, 'text': 0.05, 'kld': 0.2, 'domain': 0}
    print("loader size " + str(len(train_loader)))
    best_validate_acc = 0.000
    best_test_acc = 0.000
    best_loss = 100
    best_validate_dir = ''
    best_list = [0, 0]

    print('training model')
    adversarial = True
    # Train the Model
    ppp = 0
    # 初始化 early_stopping 对象
    patience = 20  # 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
    early_stopping = EarlyStopping(
        patience, verbose=True)  # 关于 EarlyStopping 的代码可先看博客后面的内容
    '''初始化设置寻找最优解'''
    pictuTrainLoss = []
    pictu_fnd_loss = []
    pictu_reimg_loss = []
    pictu_retext_loss = []
    pictu_kl_loss = []
    pictu_domain_loss = []
    pictuTrainACC = []
    pictuvaliLoss = []
    pictuvaliACC = []

    for epoch in range(args.num_epochs):

        p = float(epoch) / 100
        # lambd = 2. / (1. + np.exp(-10. * p)) - 1
        #学习率衰减
        lr = 0.001 / (1. + 10 * p)**0.75
        optimizer.lr = lr
        # rgs.lambd = lambd
        start_time = time.time()
        cost_vector = []
        class_cost_vector = []
        domain_cost_vector = []
        # 5个
        fnd_cost_vector = []
        recon_img_vector = []
        recon_text_vector = []
        kl_vector = []

        acc_vector = []
        valid_acc_vector = []
        test_acc_vector = []
        vali_cost_vector = []
        test_cost_vector = []
        for i, (train_data, train_labels,
                event_labels) in enumerate(train_loader):
            train_text, train_image, train_mask, train_labels, event_labels = \
                to_var(train_data[0]), to_var(train_data[1]), to_var(train_data[2]), \
                to_var(train_labels), to_var(event_labels)
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            #gh 输出
            fnd_out, recon_text, recon_img, mu, log_var, cnn_enc_img, class_outputs, domain_outputs \
                = model(train_text, train_image, train_mask)
            # gh Compute loss and accumulate the loss values
            fnd_loss, recons_loss, text_loss, kld_loss, re_all_loss = loss_function(
                train_text, cnn_enc_img, train_labels, mu, log_var, recon_text,
                recon_img, fnd_out, lambda_wts)
            # Event Loss
            domain_loss = criterion(domain_outputs, event_labels)
            # loss = class_loss + domain_loss
            # loss = re_all_loss - domain_loss
            loss = re_all_loss - 0
            loss.backward()
            optimizer.step()
            _, argmax = torch.max(fnd_out, 1)
            cross_entropy = True
            if True:
                accuracy = (train_labels == argmax.squeeze()).float().mean()
            else:
                _, labels = torch.max(train_labels, 1)
                accuracy = (
                    labels.squeeze() == argmax.squeeze()).float().mean()
            # 5个
            fnd_cost_vector.append(fnd_loss.item())
            recon_img_vector.append(recons_loss.item())
            recon_text_vector.append(text_loss.item())
            kl_vector.append(kld_loss.item())
            domain_cost_vector.append(domain_loss.item())
            # 总
            cost_vector.append(loss.item())
            acc_vector.append(accuracy.item())
        '''model.eval()是模型的某些特定层/部分的一种开关，这些层/部分在训练和推断（评估）期间的行为不同'''
        model.eval()
        validate_acc_vector_temp = []
        for i, (validate_data, validate_labels,
                event_labels) in enumerate(validate_loader):
            validate_text, validate_image, validate_mask, validate_labels, event_labels = \
                to_var(validate_data[0]), to_var(validate_data[1]), to_var(validate_data[2]), \
                to_var(validate_labels), to_var(event_labels)
            fnd_out, recon_text, recon_img, mu, log_var, cnn_enc_img, validate_outputs, domain_outputs = model(
                validate_text, validate_image, validate_mask)
            _, validate_argmax = torch.max(fnd_out, 1)
            #gh : 验证机损失函数
            fnd_loss, recons_loss, text_loss, kld_loss, re_all_loss = loss_function(
                validate_text, cnn_enc_img, validate_labels, mu, log_var,
                recon_text, recon_img, fnd_out, lambda_wts)
            validate_accuracy = (
                validate_labels == validate_argmax.squeeze()).float().mean()
            vali_cost_vector.append(re_all_loss.item())
            validate_acc_vector_temp.append(validate_accuracy.item())
        validate_acc = np.mean(validate_acc_vector_temp)
        valid_acc_vector.append(validate_acc)
        '''gh'''
        validate_loss = np.mean(vali_cost_vector)
        early_stopping(validate_loss, model)

        valid_acc_vector.append(validate_acc)
        '''gh-tensorboard'''

        model.train()
        print(
            'Epoch [%d/%d],  Loss: %.4f, Class Loss: %.4f, domain loss: %.4f, Train_Acc: %.4f,  Validate_Loss: %.4f, Validate_Acc: %.4f.'
            % (epoch + 1, args.num_epochs, np.mean(cost_vector),
               np.mean(class_cost_vector), np.mean(domain_cost_vector),
               np.mean(acc_vector), validate_loss, validate_acc))
        pictuTrainLoss.append(np.mean(cost_vector))
        pictu_fnd_loss.append(np.mean(fnd_cost_vector))
        pictu_reimg_loss.append(np.mean(recon_img_vector))
        pictu_retext_loss.append(np.mean(recon_text_vector))
        pictu_kl_loss.append(np.mean(kl_vector))
        pictu_domain_loss.append(np.mean(domain_cost_vector))

        pictuTrainACC.append(np.mean(acc_vector))
        pictuvaliACC.append(validate_acc)
        pictuvaliLoss.append(validate_loss)
        if validate_acc > best_validate_acc:
            best_validate_acc = validate_acc
            if not os.path.exists(args.output_file):
                os.mkdir(args.output_file)

            best_validate_dir = args.output_file + 'best' + '.pkl'
            torch.save(model.state_dict(), best_validate_dir)
        # 若满足 early stopping 要求
        if early_stopping.early_stop:
            print("Early stopping")
            # 结束模型训练
            break
    print("-" * 50, "训练结果", "-" * 50)
    # Test the Model
    print("-" * 50, "训练结果", "-" * 50)
    print('TrainLoss ：', pictuTrainLoss)
    print("-" * 50)
    print("fnd_loss:", pictu_fnd_loss)
    print("recon_img_loss:", pictu_reimg_loss)
    print("recon_text_loss:", pictu_retext_loss)
    print("kl_loss:", pictu_kl_loss)
    print("domain_loss:", pictu_domain_loss)
    print('TrainACC ：', pictuTrainACC)
    print('ValiACC ：', pictuvaliACC)
    print('ValiLoss ：', pictuvaliLoss)

    print('testing model')
    model = CNN_Fusion(args)
    model.load_state_dict(torch.load(best_validate_dir))
    if torch.cuda.is_available():
        model.cuda()
    '''TSEk可视化'''
    model.eval()
    test_score = []
    test_pred = []
    test_true = []

    with torch.no_grad():
        for i, (test_data, test_labels,
                event_labels) in enumerate(test_loader):
            test_text, test_image, test_mask, test_labels = to_var(
                test_data[0]), to_var(test_data[1]), to_var(
                    test_data[2]), to_var(test_labels)
            fnd_out, recon_text, recon_img, mu, log_var, cnn_enc_img, test_outputs, domain_outputs = model(
                test_text, test_image, test_mask)  # logits
            _, test_argmax = torch.max(fnd_out, 1)
            # torch.max(a, 1): 返回每一行的最大值，且返回索引:_是索引（返回最大元素在各行的列索引）。
            #all_logits.append(test_outputs)#预测向量
            #y_labels.append(test_argmax)#真真实标签
            if i == 0:
                test_score = to_np(fnd_out.squeeze())
                test_pred = to_np(test_argmax.squeeze())
                test_true = to_np(test_labels.squeeze())
            else:
                test_score = np.concatenate((test_score, to_np(fnd_out)),
                                            axis=0)
                test_pred = np.concatenate((test_pred, to_np(test_argmax)),
                                           axis=0)
                test_true = np.concatenate((test_true, to_np(test_labels)),
                                           axis=0)

    test_accuracy = metrics.accuracy_score(test_true, test_pred)
    test_f1 = metrics.f1_score(test_true, test_pred, average='macro')
    test_precision = metrics.precision_score(test_true,
                                             test_pred,
                                             average='macro')
    test_recall = metrics.recall_score(test_true, test_pred, average='macro')
    test_score_convert = [x[1] for x in test_score]
    test_aucroc = metrics.roc_auc_score(test_true,
                                        test_score_convert,
                                        average='macro')
    test_confusion_matrix = metrics.confusion_matrix(test_true, test_pred)
    print("Classification Acc: %.4f, AUC-ROC: %.4f" %
          (test_accuracy, test_aucroc))
    print("Classification report:\n%s\n" %
          (metrics.classification_report(test_true, test_pred)))
    print("Classification confusion matrix:\n%s\n" % (test_confusion_matrix))


def parse_arguments(parser):
    parser.add_argument('training_file',
                        type=str,
                        metavar='<training_file>',
                        help='')
    # parser.add_argument('validation_file', type=str, metavar='<validation_file>', help='')
    parser.add_argument('testing_file',
                        type=str,
                        metavar='<testing_file>',
                        help='')
    parser.add_argument('output_file',
                        type=str,
                        metavar='<output_file>',
                        help='')
    parse.add_argument('--static', type=bool, default=True, help='')
    parser.add_argument('--sequence_length', type=int, default=28, help='')
    parser.add_argument('--class_num', type=int, default=2, help='')
    parser.add_argument('--hidden_dim', type=int, default=32, help='')
    parser.add_argument('--embed_dim', type=int, default=32, help='')
    parser.add_argument('--vocab_size', type=int, default=300, help='')
    parser.add_argument('--dropout', type=int, default=0.5, help='')
    parser.add_argument('--filter_num', type=int, default=5, help='')
    parser.add_argument('--lambd', type=int, default=1, help='')
    parser.add_argument('--text_only', type=bool, default=False, help='')
    parser.add_argument('--d_iter', type=int, default=3, help='')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    # 100个epoch
    parser.add_argument('--num_epochs', type=int, default=100, help='')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='')
    #试试0.01
    parser.add_argument('--event_num', type=int, default=10, help='')
    parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')
    #    args = parser.parse_args()
    return parser


def get_top_post(output, label, test_id, top_n=500):
    filter_output = []
    filter_id = []
    for i, l in enumerate(label):
        # print(np.argmax(output[i]))
        if np.argmax(output[i]) == l and int(l) == 1:
            filter_output.append(output[i][1])
            filter_id.append(test_id[i])

    filter_output = np.array(filter_output)

    top_n_indice = filter_output.argsort()[-top_n:][::-1]

    top_n_id = np.array(filter_id)[top_n_indice]
    top_n_id_dict = {}
    for i in top_n_id:
        top_n_id_dict[i] = True

    pickle.dump(top_n_id_dict, open("../Data/weibo/top_n_id.pickle", "wb"))

    return top_n_id


def word2vec(post, word_id_map, W):
    word_embedding = []
    mask = []
    # length = []

    for sentence in post:
        sen_embedding = []
        seq_len = len(sentence) - 1
        mask_seq = np.zeros(args.sequence_len, dtype=np.float32)
        mask_seq[:len(sentence)] = 1.0
        for i, word in enumerate(sentence):
            sen_embedding.append(word_id_map[word])

        while len(sen_embedding) < args.sequence_len:
            sen_embedding.append(0)

        word_embedding.append(copy.deepcopy(sen_embedding))
        mask.append(copy.deepcopy(mask_seq))
        # length.append(seq_len)
    return word_embedding, mask


def re_tokenize_sentence(flag):
    tokenizer = BertTokenizer.from_pretrained(MODEL)
    tokenized_texts = []
    original_texts = flag['original_post']
    for sentence in original_texts:  # 数据中的每一个句子
        tokenized_text = tokenizer.encode(sentence)  # 编码
        tokenized_texts.append(tokenized_text)  #
    flag['post_text'] = tokenized_texts  # 覆盖


def get_all_text(train, validate, test):
    all_text = list(train['post_text']) + list(validate['post_text']) + list(
        test['post_text'])
    return all_text


# 对齐数据
def align_data(flag, args):
    text = []
    mask = []
    for sentence in flag['post_text']:  # 每一个句子拿出来
        sen_embedding = []  # 一个句子用一个list
        # 最长的max_len，初始化max——len
        mask_seq = np.zeros(args.sequence_len, dtype=np.float32)
        # mask所有的数据都覆为1.0
        mask_seq[:len(sentence)] = 1.0
        for i, word in enumerate(sentence):  # 把句子中的每一个单词进行列举
            sen_embedding.append(word)  # 嵌入向量进行append
        '''就是不满足最大长度的向量，进行补0操作，一直循环补0，补到最大长度为止'''
        while len(sen_embedding) < args.sequence_len:  #
            sen_embedding.append(0)

        text.append(copy.deepcopy(sen_embedding))
        mask.append(copy.deepcopy(mask_seq))
    flag['post_text'] = text  # 重新赋值
    flag['mask'] = mask  # mask也赋值


def load_data(args):
    # 先判断是不是只有text
    train, validate, test = process_data.get_data(args.text_only)
    print("-" * 50, "预训练模型处理中文训练集得到词向量", "-" * 50)
    re_tokenize_sentence(train)  # {字典}
    print("-" * 50, "预训练模型处理中文验证集得到词向量", "-" * 50)
    re_tokenize_sentence(validate)
    print("-" * 50, "预训练模型处理中文测试集得到词向量", "-" * 50)
    re_tokenize_sentence(test)
    print("-" * 50, "联合所有文本，找最长 max_len", "-" * 50)
    all_text = get_all_text(train, validate, test)
    max_len = len(max(all_text, key=len))
    print("max_len最长为：", max_len)
    args.sequence_len = max_len  # 将数据的sequence——len覆为最大值
    print("-" * 50, "对齐数据，填充mask：将句子变为统一长度", "-" * 50)
    align_data(train, args)
    align_data(validate, args)
    align_data(test, args)
    return train, validate, test


def transform(event):
    matrix = np.zeros([len(event), max(event) + 1])
    # print("Translate  shape is " + str(matrix))
    for i, l in enumerate(event):
        matrix[i, l] = 1.00
    return matrix


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parser = parse_arguments(parse)
    train = ''
    test = ''
    output = '/root/autodl-tmp/weight/decoder/'
    args = parser.parse_args([train, test, output])

    main(args)
