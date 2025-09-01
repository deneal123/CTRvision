import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor


class ImageClassifier(nn.Module):
    def __init__(self, model_name, num_classes, unfreeze_layers=0):
        super(ImageClassifier, self).__init__()
        self.feature_extractor = AutoImageProcessor.from_pretrained(model_name)
        self.base_model = AutoModel.from_pretrained(model_name)
        
        if unfreeze_layers == 0:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        if hasattr(self.base_model, 'config'):
            hidden_size = self.base_model.config.hidden_size
        else:
            hidden_size = 768

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        outputs = self.base_model(x)
        pooled_output = outputs.last_hidden_state[:, 0]
        return self.classifier(pooled_output)


class TabularClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(TabularClassifier, self).__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CombinedClassifier(nn.Module):
    def __init__(self, image_model_name, tabular_input_size, hidden_sizes, num_classes, unfreeze_layers=0):
        super(CombinedClassifier, self).__init__()
        self.image_feature_extractor = AutoImageProcessor.from_pretrained(image_model_name)
        self.image_base_model = AutoModel.from_pretrained(image_model_name)
        
        if unfreeze_layers == 0:
            for param in self.image_base_model.parameters():
                param.requires_grad = False

        if hasattr(self.image_base_model, 'config'):
            image_hidden_size = self.image_base_model.config.hidden_size
        else:
            image_hidden_size = 768

        self.tabular_net = TabularClassifier(tabular_input_size, hidden_sizes, hidden_sizes[-1])

        combined_input_size = image_hidden_size + hidden_sizes[-1]
        self.classifier = nn.Sequential(
            nn.Linear(combined_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, image_x, tabular_x):
        image_features = self.image_base_model(image_x).last_hidden_state[:, 0]
        tabular_features = self.tabular_net(tabular_x)
        combined = torch.cat((image_features, tabular_features), dim=1)
        return self.classifier(combined)