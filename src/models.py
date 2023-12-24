import math
import torch
from torch import nn
import torch.nn.functional as F


class GatedMultimodalLayer(nn.Module):
    """ Gated Multimodal Layer based on 'Gated multimodal networks, Arevalo1 et al.' (https://arxiv.org/abs/1702.01992) """
    def __init__(self, size_in1, size_in2, size_out):
        super().__init__()
        self.size_in1, self.size_in2, self.size_out = size_in1, size_in2, size_out

        # Weights hidden state modality 1
        weights_hidden1 = torch.Tensor(size_out, size_in1)
        self.weights_hidden1 = nn.Parameter(weights_hidden1)

        # Weights hidden state modality 2
        weights_hidden2 = torch.Tensor(size_out, size_in2)
        self.weights_hidden2 = nn.Parameter(weights_hidden2)

        # Weight for sigmoid
        weight_sigmoid = torch.Tensor(size_out*2)
        self.weight_sigmoid = nn.Parameter(weight_sigmoid)

        # initialize weights
        nn.init.kaiming_uniform_(self.weights_hidden1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weights_hidden2, a=math.sqrt(5))

        # Activation functions
        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x1, x2):
        h1 = self.tanh_f(torch.mm(x1, self.weights_hidden1.t()))
        h2 = self.tanh_f(torch.mm(x2, self.weights_hidden2.t()))
        x = torch.cat((h1, h2), dim=1)
        z = self.sigmoid_f(torch.matmul(x, self.weight_sigmoid.t()))

        return z.view(z.size()[0],1)*h1 + (1-z).view(z.size()[0],1)*h2


class AverageBERTModel(nn.Module):

    def __init__(self, hyp_params):

        super(AverageBERTModel, self).__init__()
        self.linear1 = nn.Linear(hyp_params.bert_hidden_size+hyp_params.image_feature_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(p=hyp_params.mlp_dropout)
        self.linear2 = nn.Linear(512, hyp_params.output_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, last_hidden, pooled_output, feature_images):
        
        mean_hidden = torch.mean(last_hidden, dim = 1)

        x = torch.cat((mean_hidden, feature_images), dim=1)
        x = self.act(x)
        x = self.drop1(x)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.drop1(x)

        return self.linear2(x)


class ConcatBERTModel(nn.Module):

    def __init__(self, hyp_params):

        super(ConcatBERTModel, self).__init__()
        self.linear1 = nn.Linear(hyp_params.bert_hidden_size+hyp_params.image_feature_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(p=hyp_params.mlp_dropout)
        self.linear2 = nn.Linear(512, hyp_params.output_dim)

    def forward(self, last_hidden, pooled_output, feature_images):

        x = torch.cat((pooled_output, feature_images), dim=1)
        x = self.drop1(x)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.drop1(x)

        return self.linear2(x)


class GatedAverageBERTModel(nn.Module):

    def __init__(self, hyp_params):

        super(GatedAverageBERTModel, self).__init__()
        self.gated_linear1 = GatedMultimodalLayer(hyp_params.bert_hidden_size, hyp_params.image_feature_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(p=hyp_params.mlp_dropout)
        self.linear1 = nn.Linear(512, hyp_params.output_dim)

    def forward(self, last_hidden, pooled_output, feature_images):

        mean_hidden = torch.mean(last_hidden, dim = 1)

        x = self.gated_linear1(mean_hidden, feature_images)
        x = self.bn1(x)
        x = self.drop1(x)

        return self.linear1(x)

class CSGOLOLModel(nn.Module):
    def __init__(self, hyp_params):
        super(CSGOLOLModel, self).__init__()
        self.bert = hyp_params.bert
        self.ext = hyp_params.feature_extractor
        
        self.linear1 = nn.Linear(768, 512, bias=True)
        self.ReLU = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(512, 18, bias=True)
        self.linear_ext = nn.Linear(1000, 768, bias=True)
        
    def forward(self, text_encoded, images):
        with torch.no_grad():
            outs = self.bert(**text_encoded)
        
        text = torch.mean(outs.last_hidden_state, dim=1)
        
        images = self.ext(images)
        images = self.linear_ext(images)
        
        outs = self.ReLU(gamma * text + (1-gamma) * images)
        outs = self.dropout(outs)
        outs = self.linear1(outs)
        outs = self.ReLU(outs)
        outs = self.dropout(outs)
        outs = self.linear2(outs)
        return outs
    
class RatingModel(nn.Module):
    def __init__(self, hyp_params):
        super(RatingModel, self).__init__()
        self.lin1 = nn.Linear(6040, 1024, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        self.lin2 = nn.Linear(1024, 18 , bias=True)
        
    def forward(self, ratings):
        ratings = self.relu(self.lin1(ratings))
        ratings = self.dropout(ratings)
        ratings = self.lin2(ratings)
        return ratings

class BasicModel(nn.Module):
    def __init__(self, hyp_params):
        super(BasicModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc_cnn = nn.Linear(128 * 14 * 14, 128)

        self.embed = nn.Embedding(200, 30)
        self.lstm = nn.LSTM(input_size=30, hidden_size=128, num_layers=2, batch_first=True)
        # self.fc_lstm = nn.Linear(64, 128)

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 18)
        # self.fc3 = nn.Linear(196, 18)

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, title_tensor, image_tensor, ratings):
        lstm = self.embed(title_tensor)
        lstm, (hidden, cell) = self.lstm(lstm)
        # lstm_out = self.fc_lstm(hidden[-1])
        
        images = self.conv1(image_tensor)
        images = self.pool(F.relu(self.conv2(images)))
        images = self.pool(F.relu(self.conv3(images)))
        images = torch.flatten(images, 1)
        images = self.fc_cnn(images)
        
        # out = torch.cat((hidden[-1], images), dim=1)
        out = self.dropout(hidden[-1] * 0.3 + images * 0.7)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        # out = self.relu(out)
        # out = self.dropout(out)
        # out = self.fc3(out)
        return out

class VGG_LSTMModel(nn.Module):
    def __init__(self, hyp_params):
        super(CSGOLOLModel, self).__init__()
        self.ext = hyp_params.feature_extractor
        
        self.embed = nn.Embedding(200, 30)
        self.lstm = nn.LSTM(input_size=30, hidden_size=128, num_layers=2, batch_first=True)
        # self.fc_lstm = nn.Linear(64, 128)
        
        self.linear1 = nn.Linear(1128, 512, bias=True)
        self.ReLU = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(64, 18, bias=True)
        # self.linear_ext = nn.Linear(1000, 128, bias=True)
        
    def forward(self, text_encoded, images, ratings):
        lstm = self.embed(text_encoded)
        lstm, (hidden, cell) = self.lstm(lstm)
        # lstm_out = self.fc_lstm(hidden[-1])
        
        images = self.ext(images)
        # images = self.linear_ext(images)
        
        outs = self.ReLU(torch.cat((hidden[-1], images), dim=1))
        outs = self.dropout(outs)
        outs = self.linear1(outs)
        outs = self.ReLU(outs)
        outs = self.dropout(outs)
        outs = self.linear2(outs)
        return outs