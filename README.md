# 넌씨눈을 위한 감정분석 채팅 프로그램
넌씨눈 : 눈치가 없는 사람들을 싸잡아 일컫는 말

![image](https://cdn.discordapp.com/attachments/874897301292875836/980365430302572615/2022-05-29_4.02.07.png)

## 시작하기
### requirement
```
pip install -r requirements2.txt
pip install -r requirements.txt
```
## model download

https://drive.google.com/file/d/1Qts1ONrFPvnfJR1nNj3zsuk_zRruGPjE/view?usp=sharing

clone 한 파일 안에 model폴더를 만든 후 다운로드 한 모델을 넣어주세요.

### 서버 입장
```python
python3 src/chat.py
```
![image](https://media.discordapp.net/attachments/874897301292875836/980122701790314496/2022-05-28_11.57.36.png)

http://127.0.0.1:5000 에 접속 ->

![image](https://cdn.discordapp.com/attachments/874897301292875836/980361171834118204/2022-05-29_3.45.13.png)

### 채팅방 입장
닉네임, 방이름 입력 후 입장
* 방이름이 이미 존재하는 경우 존재하는 방에 입장
* 방이름이 존재 하지 않는 경우 새로운 방 생성 후 입장

![image](https://cdn.discordapp.com/attachments/874897301292875836/980362304178757642/unknown.png)

### 감정
감정은 마지막에 입력된 텍스트의 감정을 분석하여 색깔로 표현합니다.
* 공포 : 검정
* 놀람 : 노랑
* 분노 : 빨강
* 슬픔 : 파랑
* 중립 : 초록
* 행복 : 핑크
* 혐오 : 민트초코

상대방의 감정을 생각하며 대화를 진행해 보세요!!

## 개발과정 및 기능설명

### 1. 감정분석 학습(nlp.ipynb)

Train(epochs = 10)
#### 코드
```python
max_acc = -30
train_accuracy = []
test_accuracy = []
loss_ = []
for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)
        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
            loss_.append(loss.data.cpu().numpy())
    print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
    
    train_accuracy.append(train_acc/(batch_id+1))

    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        test_acc += calc_accuracy(out, label)
    print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))

    test_accuracy.append(test_acc)

    if test_acc > max_acc:
      torch.save(model.state_dict(), '/content/drive/MyDrive/seoultech/nlp/emotion.pt')
```
#### train, test accuracy
![image](https://cdn.discordapp.com/attachments/874897301292875836/980396730157772850/KakaoTalk_Photo_2022-05-29-18-05-39.png)
#### Loss
![image](https://cdn.discordapp.com/attachments/874897301292875836/980396729910296656/KakaoTalk_Photo_2022-05-29-18-05-44.png)
#### 세부사항
best accuracy인 모델을 emotion.pt 에 저장한다.

### 2. test.py
모델을 정의하고 학습할 때의 best model을 저장하여 가중치로 사용합니다.
#### 코드
```python
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

#transformers
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=7,   ##클래스 수 조정##
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)



device = torch.device("cpu")
bertmodel, vocab = get_pytorch_kobert_model()

model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)
model.load_state_dict(torch.load('model/emotion.pt', map_location='cpu'))
model.eval()

max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 50
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

#토큰화
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

def predict(predict_sentence):

    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size)
    
    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length= valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)


        test_eval=[]
        for i in out:
            logits=i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                return (0, "공포")
            elif np.argmax(logits) == 1:
                return (1, "놀람")
            elif np.argmax(logits) == 2:
                return (2, "분노")
            elif np.argmax(logits) == 3:
                return (3, "슬픔")
            elif np.argmax(logits) == 4:
                return (4, "중립")
            elif np.argmax(logits) == 5:
                return (5, "행복")
            elif np.argmax(logits) == 6:
                return (6, "혐오")
```
#### 세부사항
* predict 함수를 이용하여 text를 입력했을 때 필요한 감정을 받아옵니다.

### 3. 채팅방 입장
![image](https://cdn.discordapp.com/attachments/874897301292875836/980366509828341860/2022-05-29_4.06.25.png)

#### 세부사항
* 닉네임과 방이름을 입력하면 입장합니다.
* 해당 방이 세션이 존재할 경우, 그곳으로 입장되며, 없을 경우 새로 생성합니다.

### 4. 채팅 및 감정 분석
![image](https://cdn.discordapp.com/attachments/874897301292875836/980367432206139392/2022-05-29_4.10.04.png)

#### 코드
```python
    @socketio.on('text', namespace='/chat')
    def text(message):
        room = session.get('room')
        emotion = test.predict(message)
        emit('message', {'msg': session.get('name') + ':' + message['msg'] + emotion}, room=room)
```
```html
socket.on('message', function(data) {
                    $('#chat').val($('#chat').val() + '[' +currentTime() + '] ' + data.msg.slice(0,-1) + '\n');
                    mess = data.msg.slice(0,-1);
                    emotion = data.msg[data.msg.length-1];
                    
                    if(emotion == 0){
                        $('h4').text("공포");
                        $('.circle').css('background','black');
                    }
                    else if(emotion == 1){
                        $('h4').text("놀람");
                        $('.circle').css('background','yellow');
                    }
                    else if(emotion == 2){
                        $('h4').text("분노");
                        $('.circle').css('background','red');
                    }
                    else if(emotion == 3){
                        $('h4').text("슬픔");
                        $('.circle').css('background','blue');
                    }
                    else if(emotion == 4){
                        $('h4').text("중립");
                        $('.circle').css('background','green');
                    }
                    else if(emotion == 5){
                        $('h4').text("행복");
                        $('.circle').css('background','pink');
                    }
                    else if(emotion == 6){
                        $('h4').text("혐오");
                        $('.circle').css('background','#81D8D0');
                    }
                    $('#chat').scrollTop($('#chat')[0].scrollHeight);
                });
```
#### 세부사항
* 메세지를 보낼 때 감정을 같이 보낸다.(import test)
* 받은 메세지에서 마지막 숫자(감정)을 떼고 메세지를 emit한다.
* 마지막 숫자(감정)에 따른 css를 적용시켜준다.

## 개발환경 및 실행 환경
Python 3.7(Mac OS)

## 데모 영상
https://user-images.githubusercontent.com/39877181/170857905-8ddb658f-d55a-405a-93c1-991de03f0892.mov

## 참고자료
[[파이썬]KoBERT로 다중 분류 모델 만들기](https://velog.io/@seolini43/KOBERT%EB%A1%9C-%EB%8B%A4%EC%A4%91-%EB%B6%84%EB%A5%98-%EB%AA%A8%EB%8D%B8-%EB%A7%8C%EB%93%A4%EA%B8%B0-%ED%8C%8C%EC%9D%B4%EC%8D%ACColab)

[Flask-Simple-Chat](https://github.com/iml1111/Flask-Simple-Chat)

