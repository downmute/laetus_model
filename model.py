import torch
import torch.nn as nn
from transformers import BertTokenizer

class ChatModel(nn.Module):
        def __init__(self, input_size=128, transformer_d=128, embedding_size=30522, embedding_dim=128, max_length=24):
            super(ChatModel, self).__init__()

            self.max_length = max_length
            self.input_size = input_size
            self.embedding_size = embedding_size
            self.embedding_dim = embedding_dim
            self.transformer_d = transformer_d
            self.embedding = nn.Embedding(num_embeddings=self.embedding_size, 
                                        embedding_dim=self.embedding_dim, 
                                        )
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.transformer_d, 
                                                            nhead=4, 
                                                            dim_feedforward=64,
                                                            batch_first=True
                                                            )
            self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer,
                                                num_layers=3,  
                                                mask_check=False
                                                )
            self.fc = nn.Sequential(
                nn.Flatten(start_dim=0),
                nn.Linear(self.max_length*self.embedding_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 5),    
                nn.Softmax()      
            )
            
        def forward(self, x):
            x = self.embedding(x)
            x = self.encoder(x)
            x = self.fc(x)
            return x
        
def main():

    tokenizer = BertTokenizer.from_pretrained("tokenizer.json")
    ids = tokenizer.encode("i spent wandering around still kinda dazed and not feeling particularly sociable but because id been in hiding for a couple for days and it was getting to be a little unhealthy i made myself go down to the cross and hang out with folks", padding='max_length', truncation=True, max_length=24)
    tokens = tokenizer.convert_ids_to_tokens(ids)
    print(tokens)

    model = ChatModel()
    ids = torch.tensor(ids)
    output = model(ids)
    print(f"Final output: {output}")
    max_value, index = torch.max(output, dim=0)
    print(f"Confidence: {max_value}, Class: {index + 1}")

if __name__ == "__main__":
    main()

#torch.save(model,"models/emotion_classifier.pt")
        

    