from collections import OrderedDict
import torch.nn as nn
from transformers import AutoModel, AutoConfig

# unfreeze_params = ['encoder.layer.11', 'pooler']
# unfreeze_params = ['encoder.layer.0']
# unfreeze_params = ['encoder.layer.11']

class Bert_model(nn.Module):
    def __init__(self, args):
        super(Bert_model, self).__init__()
        self.args = args

        self.transformer_config = AutoConfig.from_pretrained(args.transformer_name)
        self.transformer_model = AutoModel.from_pretrained(args.transformer_name, self.transformer_config)
        # for name, params in self.transformer_model.named_parameters():
        #     params.requires_grad = False
        #     for unfreeze_param in unfreeze_params:
        #         if unfreeze_param in name:
        #             params.requires_grad = True

        if self.args.linear_injection == -1:
            linear_injection = self.transformer_config.hidden_size
        else:
            linear_injection = self.args.linear_injection

        self.transformer_linear = nn.Sequential(OrderedDict([
                ('linear', nn.Linear(self.transformer_config.hidden_size, linear_injection)),
                ('layerNorm', nn.BatchNorm1d(linear_injection)),
                ('activate', nn.LeakyReLU(0.2))
            ]))

        self.classifier = nn.Linear(linear_injection, args.label_size)

    def forward(self, input_data):
        text_ids, text_masks, text_types = input_data
        bert_outputs = self.transformer_model(
            input_ids=text_ids,
            token_type_ids=text_types,
            attention_mask=text_masks)
        # (B, H)
        text_pooled_output = self.transformer_linear(bert_outputs['pooler_output'])

        logits = self.classifier(text_pooled_output)

        return logits

    def forward_with_attention(self, input_data):
        text_ids, text_masks, text_types = input_data
        bert_outputs = self.model(
            input_ids=text_ids,
            token_type_ids=text_types,
            attention_mask=text_masks,
            output_attentions=True)
        # (B, H)
        text_pooled_output = self.transformer_linear(bert_outputs['pooler_output'])
        text_attentions = bert_outputs['attentions']
        logits = self.classifier(text_pooled_output)

        return logits, text_attentions


if __name__ == '__main__':
    class Args():
        text_with_target = True
        max_tokenization_length = 128
        label_size = 2
        linear_injection = -1
        # model config
        transformer_tokenizer_name = 'model_state/vinai/bertweet-base'
        transformer_name = 'model_state/vinai/bertweet-base'

    import torch
    args = Args()
    model = Bert_model(args)
    text_ids = torch.randint(low=0, high=64001, size=[16, 128], dtype=torch.long)
    text_masks = torch.ones(size=[16, 128], dtype=torch.long)
    text_types = torch.zeros(size=[16, 128], dtype=torch.long)
    logits = model((text_ids, text_masks, text_types))
    print(f'logits.shape: {logits.shape}')