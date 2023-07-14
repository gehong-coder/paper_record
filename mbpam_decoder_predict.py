# 包
import argparse
import time, os
from src.early_stopping import *
from PIL import Image
from src import process_data_weibo2 as process_data
import copy
from random import sample
import torchvision
from torchvision import datasets, models, transforms
import torch
print(torch.version.cuda)
if (torch.cuda.is_available()):
    print("CUDA 存在")
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd

#融合 多模线性池化
from VAEMTL.pytorch_compact_bilinear_pooling import  CountSketch, CompactBilinearPooling
from VAEMTL.trans_padding import Conv1d as conv1d


#预训练模型的加载
from transformers import AutoConfig, TFAutoModel, AutoTokenizer, BertModel, BertTokenizer
MODEL = 'bert-base-chinese'
MODEL_NAME = 'bert-base-chinese'
N_LABELS = 1
import warnings
warnings.filterwarnings("ignore")
ld = {1: '虚假', 0: '真实'}  # 对应的字典



def grad_reverse(x):
    return ReverseLayerF.apply(x)

# decoder的词典大小
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
vocab = tokenizer.vocab

params_dict= {
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
    def __init__(self, vocab_size, embedding_size, max_len, latent_size, hidden_size, num_layers, bidirectional):
        super(TextDecoder, self).__init__()
        self.max_len = max_len
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.hidden_factor = (2 if bidirectional else 1) * num_layers
        self.text_decoder = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional,
                                    batch_first=True)
        self.latent2hidden = nn.Linear(latent_size, hidden_size)  ## dec text fc
        self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), vocab_size)

    def forward(self, x, max_length):
        hidden = self.latent2hidden(x)
        repeat_hidden = hidden.unsqueeze(1).repeat(1, max_length, 1)  ## repeat the hidden input to the max_len
        outputs, _ = self.text_decoder(repeat_hidden)
        outputs = outputs.contiguous()
        b, s, _ = outputs.size()
        logp = nn.functional.log_softmax(self.outputs2vocab(outputs.view(-1, outputs.size(2))), dim=1)
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
        bert_model = BertModel.from_pretrained(MODEL, output_hidden_states=True)
        self.bert_hidden_size = args.bert_hidden_dim
        for param in bert_model.parameters():
            param.requires_grad = False
        self.bertModel = bert_model
        '''卷积'''
        self.dropout = nn.Dropout(args.dropout)
        # 4卷积
        self.convs4_2 = nn.Sequential(
            nn.Conv1d(self.sequence_out, 768, 2),#32 768 191
            nn.BatchNorm1d(768),##32 768 191
            nn.LeakyReLU(),#32 768 191
            nn.MaxPool1d(self.size_pool)##32 768 63
        )
        self.convs4_3 = nn.Sequential(
            nn.Conv1d(self.sequence_out, 768, 3),
            nn.BatchNorm1d(768),
            nn.LeakyReLU(),
            nn.MaxPool1d(self.size_pool)
        )
        self.convs4_4 = nn.Sequential(
            nn.Conv1d(self.sequence_out, 768, 4),
            nn.BatchNorm1d(768),
            nn.LeakyReLU(),
            nn.MaxPool1d(self.size_pool)
        )
        self.convs4_5 = nn.Sequential(
            nn.Conv1d(self.sequence_out, 768, 5),
            nn.BatchNorm1d(768),
            nn.LeakyReLU(),
            nn.MaxPool1d(self.size_pool)
        )
        # 2卷积
        self.l2_pool = 768
        self.convs2_1 = nn.Sequential(
            conv1d(self.l2_pool, 768, 3),#torch.Size([32, 768, 251])
            nn.BatchNorm1d(self.l2_pool),
            nn.LeakyReLU(),
            nn.MaxPool1d(self.size_pool)#torch.Size([32, 768, 83])
        )
        self.convs2_2 = nn.Sequential(
            conv1d(self.l2_pool, 768, 3),
            nn.BatchNorm1d(self.l2_pool),
            nn.LeakyReLU(),
            nn.MaxPool1d(self.size_pool)
        )
        # text_append,长度不同
        self.text_flatten = 20736
        self.text_append_layer = nn.Sequential(
            nn.Linear(self.text_flatten, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(512, self.final_hid),
            nn.BatchNorm1d(32),
            nn.LeakyReLU()
        )
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
        resnet_3 = torch.nn.Sequential(*list(resnet_3.children())[:-2])  # 提取最后一层了
        self.resnet_3 = resnet_3  # 2048*7*7
          # image_append
        self.image_append_layer = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(1024, self.final_hid),
            nn.BatchNorm1d(self.final_hid),
            nn.LeakyReLU()
        )
        # region_image
        self.region = 49
        self.region_image = nn.Sequential(
            nn.Linear(2048, self.final_hid),
            nn.BatchNorm1d(self.region),
            nn.ReLU()
        )
        # attetion att_img
        self.img_dim = 32
        self.att_hid = 32
        self.head = 1
        self.img_key_layer = nn.Linear(self.img_dim, int(self.att_hid / self.head))
        self.ima_value_layer = nn.Linear(self.img_dim, int(self.att_hid / self.head))
        self.text_query = nn.Linear(self.final_hid, int(self.att_hid / self.head))
        # self.score_softmax = nn.Softmax(dim=1)
        # 注意力均值化
        self.att_average = nn.Sequential(
            nn.Linear(32, self.final_hid),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.re_Dro = nn.Sequential(
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate)
        )
        # 层激活
        self.layer_norm = nn.LayerNorm(32)

        # attention :attention text
        self.img_query = nn.Linear(self.final_hid, int(self.att_hid / self.head))
        self.text_key_layer = nn.Linear(self.bert_hidden_size, int(self.att_hid / self.head))
        self.text_value_layer = nn.Linear(self.bert_hidden_size, int(self.att_hid / self.head))
        #   soft用上一层的 att_averrage  #均值后用上面的    #层激活
        # self_attention
        self.self_img_query = nn.Linear(32, int(self.att_hid / self.head))
        self.self_img_key = nn.Linear(32, int(self.att_hid / self.head))
        self.self_img_value = nn.Linear(32, int(self.att_hid / self.head))
        # soft同上
        # self_注意力均值化
        self.self_att_average = nn.Sequential(
            nn.Linear(32, self.final_hid),
            nn.BatchNorm1d(49),
            nn.ReLU()
        )
        # 均值后数句
        self.self_re_Dro = nn.Sequential(
            nn.Linear(32, 32),
            nn.BatchNorm1d(49),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate)
        )
        # 层标准化
        self.self_layer_norm = nn.LayerNorm([49, 32])
        # flatten_self
        self.self_faltten = nn.Sequential(
            nn.Linear(49 * 32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        #文本自注意力机制-text_region
        # self_attention 32 21 768
        self.self_text_query = nn.Linear(768, int(self.att_hid / self.head))
        self.self_text_key = nn.Linear(768, int(self.att_hid / self.head))
        self.self_text_value = nn.Linear(768, int(self.att_hid / self.head))
        # soft同上
        # self_注意力均值化
        self.self_text_att_average = nn.Sequential(
            nn.Linear(32, self.final_hid),
            nn.BatchNorm1d(27),
            nn.ReLU()
        )
        # 均值后数句
        self.self_text_re_Dro = nn.Sequential(
            nn.Linear(32, 32),
            nn.BatchNorm1d(27),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate)
        )
        # 层标准化
        self.self_text_layer_norm = nn.LayerNorm([27, 32])
        # flatten_self
        self.self_text_faltten = nn.Sequential(
            nn.Linear(27 * 32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        #image
        self.multi = nn.Linear(768, 32)
        self.merge = nn.Linear(1024,32)
        self.mcb = CompactBilinearPooling(27, 49, 32)
        # 融合激活
        self.merge_feature = nn.Sequential(
            nn.Linear(160, self.final_hid),
            nn.Dropout(self.drop_rate)
        )
        # Class Classifier
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(self.hidden_size, 2))
        self.class_classifier.add_module('c_softmax', nn.Softmax(dim=1))

        # Event Classifier
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(self.hidden_size, self.hidden_size))
        self.domain_classifier.add_module('d_relu1', nn.LeakyReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(self.hidden_size, self.event_num))
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
                            nn.Linear( self.latent_dim , self.fnd_fc1),
                            nn.Tanh(),
                            nn.Linear(self.fnd_fc1, self.fnd_fc2),
                            nn.Tanh(),
                            nn.Linear(self.fnd_fc2, 2),
                            nn.Sigmoid()
        )
        #decoder用到上面的结构
        self.text_decoder = TextDecoder(params_dict['vocab_size'], params_dict['embedding_size'], params_dict['max_len'], params_dict['latent_dim'], params_dict['hidden_size'], params_dict['num_layers'], params_dict['bidirectional'])
        self.visual_decoder = VisualDecoder(params_dict['latent_dim'], params_dict['dec_fc_img_1'], params_dict['enc_img_dim'])

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

    def forward(self, text, image, mask):# torch.Size([32, 192]) torch.Size([32, 3, 224, 224]) torch.Size([32, 192])
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
        sequence_output = torch.cat((all_hidden_state[-1], all_hidden_state[-2], all_hidden_state[-3], all_hidden_state[-4],all_hidden_state[-5]), 2)
        #print(sequence_output.shape)    #torch.Size([32, 192, 3840])
        # 由于torch卷积的原因，需要将max_len和词向量维度进行转变使用permute函数
        sequence_output = sequence_output.permute(0, 2, 1)  #torch.Size([32, 3840, 192])
        #print(sequence_output.shape)
        '''四卷积'''
        convs = []
        l_pool_2 = self.convs4_2(sequence_output)   #torch.Size([32, 768, 63])
        #print(l_pool_2.shape)
        convs.append(l_pool_2)
        l_pool_3 = self.convs4_3(sequence_output)   #torch.Size([32, 768, 63])
        #print(l_pool_3.shape)
        convs.append(l_pool_3)
        l_pool_4 = self.convs4_4(sequence_output)   #torch.Size([32, 768, 63])
        #print(l_pool_4.shape)
        convs.append(l_pool_4)
        l_pool_5 = self.convs4_5(sequence_output)   #torch.Size([32, 768, 62])
        #print(l_pool_5.shape)
        convs.append(l_pool_5)
        '''拼接4卷积-convs'''
        l2_pool = torch.cat(convs, dim=2)
        #print(l2_pool.shape)    #torch.Size([32, 768, 251])
        '''两卷积'''
        # 卷积 - 批标准化 - 激活- 池化
        l2_pool = self.convs2_1(l2_pool)    #torch.Size([32, 768, 83])
        #print(l2_pool.shape)
        l2_pool = self.convs2_2(l2_pool)    #torch.Size([32, 768, 27])

        #print(l2_pool.shape)
        '''卷积完成，将词向量的维度和max——len进行调换位置-TM：【batch，区域数，向量维度】'''
        Tm = l2_pool.permute(0, 2, 1)   #torch.Size([32, 27, 768])
        #print(Tm.shape)batch,chanel,dim
        '''展平,（batch 不展平）此时后面的两个维度交不交换没有必要性，因为后面要进行flatten操作'''
        text = torch.flatten(Tm, start_dim=1, end_dim=2)    #torch.Size([32, 20736])
        #print(text.shape)

        '''text双流1: 词特征tm'''
        #multi_text = self.multi(text)   #torch.Size([32, 1568])
        multi_text = self.multi(Tm)#32 27 768--32 27 32
        #print(multi_text.shape)
        multi_text = multi_text.permute(0,2,1)#batch,dim,chanel
        #print(multi_text.shape)
        merge = []
        '''text双流2:text_append[batch_size,32],直接拼接 句特征'''
        text_append = self.text_append_layer(text)
        #print(text_append.shape)    #torch.Size([32, 32])

        merge.append(text_append)

        '''image_双流1：image_append——倒数一层vgg'''
        image_1 = self.resnet_1(image)  #   torch.Size([32, 2048])
        cnn_enc_img = image_1
        #print(image_1.shape)
        image_append = self.image_append_layer(image_1) #torch.Size([32, 32])
        #print(image_append.shape)
        merge.append(image_append)
        '''image双柳2：区域image——倒数3层'''
        image_3 = self.resnet_3(image)    # torch.Size([32, 2048, 7, 7])
        #print(image_3.shape)
        image_3 = image_3.permute(0, 2, 3, 1)  # torch.Size([32, 7, 7, 2048])
        image_3 = torch.reshape(image_3, ((image_3.shape[0], -1, image_3.shape[-1])))
        #print(image_3.shape)#torch.Size([32, 49, 2048])
        image = self.region_image(image_3)#torch.Size([32, 49, 32])
        #print(image.shape)
        Im = image
        #multi_image = torch.flatten(Im, start_dim=1, end_dim=2)  #torch.Size([32, 1568])
        multi_image = image.permute(0,2,1)#32 32,49
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
        Merge_feture= torch.flatten(Merge_feture, start_dim=1, end_dim=2)
        Merge_feture = self.merge(Merge_feture)#
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
                score = torch.tensordot(self_img_query, self_img_key, dims=([2], [2]))  # torch.Size([32, 32, 21])
                # print(score.shape)  # torch.Size([32, 49,32, 49])
                '''改变合并方法'''
                score = torch.stack([score[i, :, i, :] for i in range(len(score))])
                # print(score.shape)  # torch.Size([32, 49,49])
                # print("score.shape")
                '''公式第二步,score/根号下d 维度-维度不匹配的广播机制'''
                # 如果是y一个数，x所有元素除以y
                score = torch.div(score, np.sqrt(att_hid / head))  # [32,49,49]
                score = F.leaky_relu(score)
                '''公式第三步-score * v'''
                # score：torch.Size([32, 49,49]) image_value = torch.Size([32, 49, 32])
                attention = torch.tensordot(score, self_img_value, dims=([2], [1]))
                # print(attention.shape)  # torch.Size([32, 49, 32,32])
                attention = torch.stack([attention[i, :, i, :] for i in range(len(attention))])
                # print(attention.shape)  #   torch.Size([32, 49, 32])
                '''得出询问后的自己后的 att'''
                self_att_img = attention
                '''将注意力均值化-无法平均-也就是多个列表，对应的元素相加求平均在进行成为当前元素'''
                self_att_img21 = self.self_att_average(self_att_img)
                self_att_img22 = self.self_att_average(self_att_img)
                self_att_img23 = self.self_att_average(self_att_img)
                self_att_img24 = self.self_att_average(self_att_img)
                # 均值
                self_att_img2 = self_att_img21.add(self_att_img22).add(self_att_img23).add(self_att_img24)
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
        self_att_img = torch.flatten(self_att_img, start_dim=1, end_dim=2)  # end_dim=-1
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
                self_text_query = self.self_text_query(in_Self_TEXT_query)##torch.Size([32, 27, 32])
                self_text_key = self.self_text_key(in_Self_TEXT_key)#torch.Size([32, 27, 32])
                self_text_value = self.self_text_value(in_Self_TEXT_key)#torch.Size([32, 27, 32])
                '''计算sccore Q*K-公式第一步'''
                score = torch.tensordot(self_text_query, self_text_key, dims=([2], [2]))  # torch.Size([32, 32, 21])
                #print(score.shape)  #torch.Size([32, 27, 32, 27])
                '''改变合并方法'''
                score = torch.stack([score[i, :, i, :] for i in range(len(score))])
                #print(score.shape)  #torch.Size([32, 27, 27])
                '''公式第二步,score/根号下d 维度-维度不匹配的广播机制'''
                # 如果是y一个数，x所有元素除以y
                score = torch.div(score, np.sqrt(att_hid / head))
                score = F.leaky_relu(score)
                '''公式第三步-score * v'''
                attention = torch.tensordot(score, self_text_value, dims=([2], [1]))
                attention = torch.stack([attention[i, :, i, :] for i in range(len(attention))])
                '''得出询问后的自己后的 att'''
                self_text_att_img = attention
                '''将注意力均值化-无法平均-也就是多个列表，对应的元素相加求平均在进行成为当前元素'''
                self_text_att21 = self.self_text_att_average(self_text_att_img)
                self_text_att22 = self.self_text_att_average(self_text_att_img)
                self_text_att23 = self.self_text_att_average(self_text_att_img)
                self_text_att24 = self.self_text_att_average(self_text_att_img)
                # 均值
                self_text_att2 = self_text_att21.add(self_text_att22).add(self_text_att23).add(self_text_att24)
                self_text_att2 = torch.div(self_text_att2, 4)
                '''均值后数据'''
                self_text_att2 = self.self_text_re_Dro(self_text_att2)
                '''将注意力后的数据相加:image_+self_att_img'''
                self_text_att = torch.add(self_text_query, self_text_att2) # [32,27,32]
                '''层标准化'''
                self_text_att = self.self_text_layer_norm(self_text_att)
                '''不能将他直接merge进去因为，维度不同，faltten之后进行输入'''
                in_Self_TEXT_query = self_text_att
                in_Self_TEXT_key = self_text_att
        '''将后面的两维度flatten，变成二维'''
        self_text_att = torch.flatten(self_text_att, start_dim=1, end_dim=2)  # end_dim=-1
        self_text_att = self.self_text_faltten(self_text_att)
        '''此时向量的维度是正常的'''
        #print(self_text_att.shape)  #torch.Size([32, 32])
        merge.append(self_text_att)
        #print("-" * 50, "结束融合", "-" * 50)

        '''1。共有5部分数据- 32*32 ,32-160'''
        feature_merge = torch.cat(merge, dim=1)  # 32, 160

        # 改进点3：基于基于自编码器进行多任务学习

        """2。参数的重构VAR"""
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(feature_merge)# 32 32
        #print(mu.shape) # 32 32
        log_var = self.fc_var(feature_merge)# 32 32
        #print(log_var.shape)
        z = self.reparameterize(mu, log_var) #32 32
        #print(z.shape)

        """decoder重建的text和图像"""

        recon_text, recon_img = self.decode(z, text_decoder.shape[1])
        # print(recon_text.shape)#torch.Size([32, 192, 21128]), 因为重建的文本，每个都是单词
        # print(recon_img.shape)# torch.Size([32, 4096])

        """3。虚假新闻分类器"""
        fnd_out = self.fnd_module(z)#torch.Size([32, 2])
        # fnd_out = self.merge_feature(feature_merge)#torch.Size([32, 2])
        # print(fnd_out.shape)#
        feature_merge = self.merge_feature(feature_merge)
        fnd_out = self.class_classifier(feature_merge)
        class_output = fnd_out
        """4。域分类器"""
        reverse_feature = grad_reverse(feature_merge)
        domain_output = self.domain_classifier(reverse_feature)
        #print(cnn_enc_img.shape)#torch.Size([32, 2048])

        return [fnd_out, recon_text, recon_img, mu, log_var, cnn_enc_img, class_output, domain_output]


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)



def parse_arguments(parser):
    parser.add_argument('training_file', type=str, metavar='<training_file>', help='')
    # parser.add_argument('validation_file', type=str, metavar='<validation_file>', help='')
    parser.add_argument('testing_file', type=str, metavar='<testing_file>', help='')
    parser.add_argument('output_file', type=str, metavar='<output_file>', help='')
    parser.add_argument('--static', type=bool, default=True, help='')
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

# 首先看怎么加载模型
"""提前加载模型"""
parse = argparse.ArgumentParser()
parser = parse_arguments(parse)
train = ''
test = ''
output = '/tmp/pycharm_project_766/src/weight'
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
args = parser.parse_args([train, test, output])
print("----虚假新闻检测模型加载中-----")
best_validate_dir = '/root/autodl-tmp/weight/decoder/best.pkl'
print("----模型加载中-----")
model = CNN_Fusion(args)
model.eval()
print("----模型参数加载中-----")
model.load_state_dict(torch.load(best_validate_dir))
print("-----虚假新闻检测模型加载完毕----")
sequence_len = 192


def preprocess(text,img):
    tokenized_text = tokenizer.encode(text)  # 编码
    sen_embedding = []  # 一个句子用一个list
    # 最长的max_len，初始化max——len
    mask_seq = np.zeros(sequence_len, dtype=np.float32)
    # mask所有的数据都覆为1.0，其余的都为0
    mask_seq[:len(text)] = 1.0
    for i, word in enumerate(tokenized_text):  # 把句子中的每一个单词进行列举
        sen_embedding.append(word)  # 嵌入向量进行append
    '''就是不满足最大长度的向量，进行补0操作，一直循环补0，补到最大长度为止'''
    while len(sen_embedding) < sequence_len:  #
        sen_embedding.append(0)
    print('原始文本:', text)
    print("mask_seq:", mask_seq)
    print("sen_embedding:", sen_embedding)
    sen_embedding = torch.from_numpy(np.array(sen_embedding))
    if len(sen_embedding) >= 192:
        sen_embedding = sen_embedding[:192]
    sen_embedding = sen_embedding.unsqueeze(0)
    mask_seq = torch.from_numpy(np.array(mask_seq))
    mask_seq = mask_seq.unsqueeze(0)
    print('sentence的shape:', sen_embedding.size())
    print('mask_seq的shape:', mask_seq.size())
    '''flask调用时覆盖'''
    img = img.unsqueeze(0)
    print("img的shape:",img.size())
    return sen_embedding, img, mask_seq


#使用模型进行预测的过程
def predict(text,img):
    sen_embedding, img, mask_seq = preprocess(text,img)
    with torch.no_grad():
        print("----开始预测------")
        test_outputs = model(sen_embedding, img, mask_seq)  # logits
        _, predict = torch.max(test_outputs[0], 1)
    # print("虚假新闻预测结果：",ld[predict.numpy().tolist()[0]], type(predict.numpy().tolist()))  # 输出对应的标签
    # print("虚假新闻预测结果",predict)
    return ld[predict.numpy().tolist()[0]]

import os
array_of_img = [] # this if for store all of the image data
# this function is for read image,the input is directory name
def read_directory(directory_name):
    file_name = os.listdir(directory_name)
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(directory_name):
        #print(filename) #just for test
        #img is used to store the image data
        img = cv2.imread(directory_name + "/" + filename)
        array_of_img.append(img)
        #print(img)
        print(array_of_img)
    return file_name,array_of_img

from sklearn import metrics
def calcu_acc():
    real_label = []
    for i in range(660):
        real_label.append(0)
    pre=['真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '虚假', '真实', '真实', '真实', '真实', '虚假', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '虚假', '真实', '真实', '真实', '真实', '虚假', '虚假', '真实', '真实', '虚假', '真实', '虚假', '虚假', '虚假', '虚假', '真实', '真实', '虚假', '真实', '虚假', '真实', '虚假', '虚假', '虚假', '真实', '虚假', '真实', '虚假', '真实', '真实', '虚假', '虚假', '虚假', '虚假', '真实', '真实', '虚假', '真实', '真实', '真实', '虚假', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '虚假', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '虚假', '真实', '真实', '真实', '真实', '真实', '虚假', '真实', '虚假', '虚假', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '虚假', '真实', '虚假', '真实', '真实', '真实', '真实', '真实', '虚假', '真实', '真实', '真实', '真实', '真实', '虚假', '虚假', '真实', '真实', '真实', '真实', '真实', '真实', '虚假', '真实', '虚假', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '虚假', '真实', '虚假', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '虚假', '真实', '虚假', '真实', '真实', '虚假', '虚假', '真实', '虚假', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '虚假', '虚假', '真实', '虚假', '真实', '真实', '真实', '真实', '真实', '虚假', '虚假', '虚假', '虚假', '真实', '真实', '真实', '真实', '真实', '虚假', '真实', '真实', '虚假', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '虚假', '真实', '虚假', '真实', '真实', '虚假', '真实', '真实', '真实', '真实', '虚假', '虚假', '虚假', '虚假', '真实', '真实', '虚假', '真实', '虚假', '虚假', '虚假', '真实', '真实', '虚假', '真实', '真实', '真实', '真实', '真实', '虚假', '真实', '真实', '虚假', '真实', '虚假', '虚假', '真实', '真实', '虚假', '真实', '虚假', '真实', '虚假', '真实', '虚假', '虚假', '真实', '真实', '真实', '真实', '虚假', '真实', '虚假', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '虚假', '虚假', '真实', '虚假', '真实', '虚假', '真实', '真实', '真实', '虚假', '真实', '真实', '虚假', '真实', '真实', '真实', '虚假', '真实', '虚假', '真实', '虚假', '真实', '真实', '真实', '虚假', '虚假', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '虚假', '真实', '真实', '虚假', '虚假', '真实', '真实', '真实', '真实', '真实', '真实', '虚假', '虚假', '真实', '真实', '真实', '虚假', '真实', '真实', '真实', '真实', '真实', '虚假', '真实', '虚假', '真实', '虚假', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '虚假', '真实', '真实', '真实', '真实', '虚假', '真实', '虚假', '真实', '虚假', '真实', '真实', '虚假', '虚假', '真实', '虚假', '虚假', '真实', '真实', '真实', '虚假', '真实', '虚假', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '虚假', '真实', '真实', '虚假', '虚假', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '虚假', '真实', '真实', '真实', '真实', '虚假', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '虚假', '真实', '真实', '真实', '真实', '虚假', '虚假', '真实', '真实', '虚假', '真实', '虚假', '虚假', '虚假', '虚假', '真实', '真实', '虚假', '真实', '虚假', '真实', '虚假', '虚假', '虚假', '真实', '虚假', '真实', '虚假', '真实', '真实', '虚假', '虚假', '虚假', '虚假', '真实', '真实', '虚假', '真实', '真实', '真实', '虚假', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '虚假', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '虚假', '真实', '真实', '真实', '真实', '真实', '虚假', '真实', '虚假', '虚假', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '虚假', '真实', '虚假', '真实', '真实', '真实', '真实', '真实', '虚假', '真实', '真实', '真实', '真实', '真实', '虚假', '虚假', '真实', '真实', '真实', '真实', '真实', '真实', '虚假', '真实', '虚假', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '虚假', '真实', '虚假', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '虚假', '真实', '虚假', '真实', '真实', '虚假', '虚假', '真实', '虚假', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '虚假', '虚假', '真实', '虚假', '真实', '真实', '真实', '真实', '真实', '虚假', '虚假', '虚假', '虚假', '真实', '真实', '真实', '真实', '真实', '虚假', '真实', '真实', '虚假', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '虚假', '真实', '虚假', '真实', '真实', '虚假', '真实', '真实', '真实', '真实', '虚假', '虚假', '虚假', '虚假', '真实', '真实', '虚假', '真实', '虚假', '虚假', '虚假', '真实', '真实', '虚假', '真实', '真实', '真实', '真实', '真实', '虚假', '真实', '真实', '虚假', '真实', '虚假', '虚假', '真实', '真实', '虚假', '真实', '虚假', '真实', '虚假', '真实', '虚假', '虚假', '真实', '真实', '真实', '真实', '虚假', '真实', '虚假', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '真实', '虚假', '虚假', '真实', '虚假']
    pred = []
    for i in pre:
        if i=="真实":
            pred.append(0)
        else:
            pred.append(1)
    print(metrics.classification_report(real_label, pred))


if __name__=="__main__":
    # data_transforms = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    # text = "jddj"
    # img = torch.rand(3,224,224)
    # # path = "/Users/gehong/Documents/Transformer实战/nlp-notebook-master/爬虫/最终数据/最终版1123.csv"
    # path = "./爬取数据/news_爬取.csv"
    # img_path = '/tmp/pycharm_project_815/src/sci/爬取数据/image'
    # df = pd.read_csv(path,sep='\t')
    # l = len(df)
    # file_name = os.listdir(img_path)
    # result = []
    # for index in range(l):
    #     sentence = df["content"][index]
    #     img_str = df["name_str"][index]
    #     if img_str in file_name:
    #         img = Image.open(img_path +'/'+img_str).convert('RGB')
    #         img = data_transforms(img)
    #         predict_label = predict(sentence,img)
    #         result.append(predict_label)
    # print(result)
    # calcu_acc()
    text = ''
    # img = torch.rand(3, 224, 224)
    # predict(text,img)
