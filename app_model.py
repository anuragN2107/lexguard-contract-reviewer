import torch
import torch.nn as nn
from transformers import AutoModel

class LegalBertCNNClassifier(nn.Module):
    def __init__(self, bert_model_name="nlpaueb/legal-bert-base-uncased", num_classes=4, num_filters=100, filter_sizes=[3, 4, 5]):
        super(LegalBertCNNClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        embedding_dim = self.bert.config.hidden_size # 768
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = bert_outputs.last_hidden_state
        hidden_states = hidden_states.permute(0, 2, 1) # [Batch, Embed, Seq]
        
        pooled_outputs = []
        for conv in self.convs:
            x = torch.relu(conv(hidden_states))
            x = torch.max(x, dim=2)[0]
            pooled_outputs.append(x)
            
        cat = self.dropout(torch.cat(pooled_outputs, dim=1))
        return self.fc(cat)