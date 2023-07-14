#'''1.包'''
from PIL import Image
from transformers import AutoConfig, TFAutoModel, AutoTokenizer, BertModel, BertTokenizer
from flask import Flask, request,render_template
import  numpy as np
import torch
from torchvision import datasets, models, transforms
import base64

# 2.初始化模型， 避免在函数内部初始化，耗时过长
# from src.sci import weibopredict
# MODEL_NAME = 'bert-base-chinese'
MODEL_NAME = 'bert-base-chinese'


tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
import mbpam_decoder_predict


# 3.初始化flask
app = Flask(__name__)

#4. 数据库存储
from model import *

#4.启动服务后会进到一个起始页
'''起始页'''
@app.route('/', methods=['GET', 'POST'])
def log():
    return render_template('boot.html')

#5.上传数据
@app.route('/up_photo', methods=['GET','post'])
def up_photo():
    '''图像处理预先加载'''
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if request.method == 'POST':  # 判断是否是 POST 请求 表单请求
        # 5.1获取文本内容，并预处理
        sentence = request.form.get('text')  # 传入表单对应输入字段的 name 值
        print("ssss:",sentence)
        tokenized_text = tokenizer.encode(sentence)  # 编码
        sequence_len = 192
        sen_embedding = []  # 一个句子用一个list
        mask_seq = np.zeros(sequence_len, dtype=np.float32)
        mask_seq[:len(sentence)] = 1.0 # mask所有的数据都覆为1.0
        for i, word in enumerate(tokenized_text):  # 把句子中的每一个单词进行列举
            sen_embedding.append(word)  # 嵌入向量进行append
        '''不满足最大长度的向量，进行补0操作，一直循环补0，补到最大长度为止'''
        while len(sen_embedding) < sequence_len:
            sen_embedding.append(0)
        print('原始文本',sentence)
        print(mask_seq)
        print(sen_embedding)
        sen_embedding = torch.from_numpy(np.array(sen_embedding))
        mask_seq = torch.from_numpy(np.array(mask_seq))
        print(sen_embedding)
        # 5.2获取图像内容，进行预处理
        img1 = request.files['photo']
        print(type(img1))
        img = Image.open(img1.stream)
        print(sentence, img)
        im = img.convert('RGB')
        '''图像处理成tensor'''
        im = data_transforms(im)
        print(im.shape)
        context = {
            'text':sentence,
            'tensor_mask':mask_seq,
            'tensor_text': sen_embedding,
            'image':img1,
            'tensor_image': im,
        }
        # 将接受的到数据处理完成的送入到展示页面
    return render_template('process.html',context=context)


# 预测新闻
@app.route('/predict', methods=['GET', 'POST'])
def predict_news():
    if request.method == 'POST':  # 判断是否是 POST 请求        # 获取表单数据
        sentence = request.form.get('text')  # 传入表单对应输入字段的 name 值
        print('原始文本',sentence)
        '''图像'''
        img1 = request.files['photo']

        #存到数据库
        img2 = request.files['photo'].read()
        img3 = base64.b64encode(img2)
        # print(img3)
        print(type(img3))
        img = Image.open(img1.stream)
        print(sentence, img)
        im = img.convert('RGB')
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])#'''图像处理成tensor'''
        im = data_transforms(im)
        print(im.shape)
        # 将预处理完成的数据，送入模型进行检测。
        result = mbpam_decoder_predict.predict(text=sentence,img=im)        #会返回结果
        context = {
            'text':sentence,
            'image':img,
            'result_label':result
        }
        print(result)

        # 数据库存储
        news_sql = news_(text=sentence,image=img3,result=result)
        db.session.add(news_sql)#添加数据
        db.session.commit()#数据提交

    return render_template('predict2.html',context=context)


#将数据加载到html页面进行展示
@app.route('/check_data', methods=['GET', 'post'])
def check_data():
    '''图像处理预先加载'''
    print('ok')
    all_news = news_.query.all()
    print(all_news)
    for i in all_news:
        print(i.text)
    print("数据")
    context = all_news
    return render_template('all_data.html',context=context)

    # python3 -m flask run
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6006, debug=True, use_reloader=False)
