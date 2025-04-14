import os
import torch
import numpy as np
import pickle
from tqdm import tqdm
from transformers import Wav2Vec2Processor
import librosa
from collections import defaultdict
from torch.utils import data 
from transformers import BertTokenizer, BertModel
class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data,data_type="train",read_audio=False):
        self.data = data
        self.len = len(self.data)
        self.data_type = data_type
        self.read_audio = read_audio

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim

        audio = self.data[index]["audio"]
        text = self.data[index]["text_emb"]
        return torch.FloatTensor(audio),torch.FloatTensor(text)
 

    def __len__(self):
        return self.len
def read_txt_files(file_path):
    output=[]
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
        # 处理每一行
            output.append(line)  # 使用strip()去除每行的换行符
    return output
def read_data(args):
    print("Loading data...")
    data = defaultdict(dict)
    train_data = []
    processor = Wav2Vec2Processor.from_pretrained(args.audio_processor)
    tokenizer = BertTokenizer.from_pretrained(args.BERT_tokenizer)
    bert = BertModel.from_pretrained(args.BERT_model)
    for f in tqdm(os.listdir(args.data_root)):
        file_path = os.path.join(args.data_root, f)
        for fd in os.listdir(file_path):
            data_file_path =  os.path.join(file_path,fd)
            text_path = os.path.join(data_file_path ,f + '-' + fd +'.trans.txt')
            txt = read_txt_files(text_path)
            for i in range(len(txt)):
                data_name, data_text = txt[i].split(' ', 1)
                data_path = os.path.join(data_file_path, data_name + '.flac')
                speech_array, sampling_rate = librosa.load(data_path, sr=16000)
                input_values = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
                key = f+'-'+ fd 
                data[key]["audio"] = input_values
                encoded_input = tokenizer(data_text, return_tensors='pt')
                text_emb = bert(**encoded_input).last_hidden_state[:,0,:][0].cpu().detach().numpy()
                data[key]["text_emb"] = text_emb    

    for k, v in data.items():
            train_data.append(v)
        
    print('Loaded data: Train-{}'.format(len(train_data)))
    return train_data

def get_dataloaders(args):
    dataset = {}
    train_data = read_data(args)
    train_data = Dataset(train_data,"train",args.read_audio)
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    return dataset

if __name__ == "__main__":
    get_dataloaders()