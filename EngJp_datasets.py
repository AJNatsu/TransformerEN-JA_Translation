import torch
import torch.nn as nn
from torch.utils.data import Dataset
#initialize dataset


class EngJpDataset(Dataset):

    def __init__(self, data, tokenizer_src, tokenizer_trg, src_lang, trg_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.data = data
        self.tokenizer_src = tokenizer_src
        self.tokenizer_trg = tokenizer_trg
        self.src_lang = src_lang
        self.trg_lang = trg_lang

        #gets the token ID for a start-of-sequence ([SOS]) token, converts it to a tensor
        self.sos_token = torch.tensor([tokenizer_src.convert_tokens_to_ids("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.convert_tokens_to_ids("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.convert_tokens_to_ids("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_target_pair = self.data[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        trg_text = src_target_pair['translation'][self.trg_lang]

        # split the sentence into each word, convert each word into its corresponding number in the vocab
        enc_input_tokens = self.tokenizer_src.encode(src_text)
        dec_input_tokens = self.tokenizer_trg.encode(trg_text)

        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        # We will only add <s>, and </s> only on the label
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Add SOS and EOS to the source text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],

        )
        #Add SOS to the decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],


        )
        #Add EOS to the label ( what we expect as output from the decoder
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],


        )

        #Double-check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  #(seq_len)
            "decoder_input": decoder_input, #(seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), #(1,1,seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "trg_text": trg_text,
        }


def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0
