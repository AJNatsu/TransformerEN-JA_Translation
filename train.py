import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from EngJp_datasets import EngJpDataset, causal_mask
from model import build_transformer
from config import get_config, get_weights_file_path, latest_weights_file_path
#hugging face
from datasets import load_dataset
from tokenizers import Tokenizer

from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from transformers import BertJapaneseTokenizer
from transformers import GPTNeoXJapaneseTokenizer

from torch.utils.tensorboard import SummaryWriter
import torchmetrics

import warnings
from pathlib import Path
from tqdm import tqdm


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_trg, max_len, device):
    sos_idx = tokenizer_trg.convert_tokens_to_ids("[SOS]")
    eos_idx = tokenizer_trg.convert_tokens_to_ids("[EOS]")

    #precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, source_mask)
    #Initializae the decoder input with the sos taken
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        #Build mask for the target(decoder input)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        #Calculate the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        #Get the next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_data, tokenizer_src, tokenizer_trg, max_len, device, print_msg, global_step, writer, num_examples =2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    total_val_loss = 0
    num_batches = 0
    #Size of the control window(just use a default value
    console_width = 80
    with torch.no_grad():
        for batch in validation_data:
            count +=1
            encoder_input = batch['encoder_input'].to(device)  # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device)  # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)  # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)  # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask)  # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input,
                                          decoder_mask)  # (B, seq_len, d_model)
            proj_output = model.project(decoder_output)  # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device)  # (B, seq_len)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_trg, max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['trg_text'][0]
            model_out_text = tokenizer_trg.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.convert_tokens_to_ids('[PAD]'))
            val_loss = loss_fn(proj_output.view(-1, tokenizer_trg.vocab_size), label.view(-1))
            total_val_loss += val_loss.item()
            num_batches += 1
            #print to the console
            print_msg('-'*console_width)
            print_msg(f'SOURCE:{source_text}')
            print_msg(f'TARGET:{target_text}')
            print_msg(f'PREDICTED:{model_out_text}')

            if count == num_examples:
                break

    if writer:
        # Evaluate the character error rate
        # Compute the char error rate
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()

    avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0
    return avg_val_loss


def get_all_sentences(data, lg):
    for item in data:
        yield item['translation'][lg]


def get_or_build_tokenizer(config, data, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = GPTNeoXJapaneseTokenizer.from_pretrained("abeja/gpt-neox-japanese-2.7b")
        tokenizer.save_pretrained(str(tokenizer_path))
    else:
        tokenizer = GPTNeoXJapaneseTokenizer.from_pretrained(str(tokenizer_path))
    return tokenizer



def get_data(config):
    # It only has the train split, so we divide it overselves
    data_raw = load_dataset(f"{config['datasource']}", f"{config['lg_src']}-{config['lg_trg']}", split='test')

    print(data_raw[0])
    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, data_raw, config['lg_src'])
    tokenizer_trg = get_or_build_tokenizer(config, data_raw, config['lg_trg'])

    print(tokenizer_src.encode("This is a sample sentence."))
    print(tokenizer_trg.encode("これはサンプルの文章です。"))
    # Print tokenizer configuration
    # print(tokenizer_src)
    # print(tokenizer_trg)

    # Keep 90% for training, 10% for validation
    train_data_size = int(0.9 * len(data_raw))
    val_data_size = len(data_raw) - train_data_size
    train_data_raw, val_data_raw = random_split(data_raw, [train_data_size, val_data_size])

    train_data = EngJpDataset(train_data_raw, tokenizer_src, tokenizer_trg, config['lg_src'], config['lg_trg'], config['seq_len'])
    val_data = EngJpDataset(val_data_raw, tokenizer_src, tokenizer_trg, config['lg_src'], config['lg_trg'], config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_trg = 0

    # Encode a sample sentence and print the token IDs and length
    # sample_src_sentence = data_raw[0]['translation'][config['lg_src']]
    # sample_trg_sentence = data_raw[0]['translation'][config['lg_trg']]
    # src_ids = tokenizer_src.encode(sample_src_sentence).ids
    # trg_ids = tokenizer_trg.encode(sample_trg_sentence).ids
    # print(f'Sample Source Sentence: {sample_src_sentence}')
    # print(f'Token IDs: {src_ids}, Length: {len(src_ids)}')
    # print(f'Sample Target Sentence: {sample_trg_sentence}')
    # print(f'Token IDs: {trg_ids}, Length: {len(trg_ids)}')

    for item in data_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lg_src']])
        trg_ids = tokenizer_trg.encode(item['translation'][config['lg_trg']])
        max_len_src = max(max_len_src, len(src_ids))
        max_len_trg = max(max_len_trg, len(trg_ids))


    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_trg}')

    # Print the first 20 items of the vocabulary
    # src_vocab = tokenizer_src.get_vocab()
    # trg_vocab = tokenizer_trg.get_vocab()
    # print("Source Vocabulary: ", list(src_vocab.keys())[:20])
    # print("Target Vocabulary: ", list(trg_vocab.keys())[:20])

    train_dataloader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_trg


def get_model(config, vocab_src_len, vocab_trg_len):
    model = build_transformer(vocab_src_len, vocab_trg_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model


def train_model(config):
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)

    train_losses = []
    val_losses = []
    # Make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_trg = get_data(config)
    model = get_model(config, tokenizer_src.vocab_size, tokenizer_trg.vocab_size).to(device)
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config,preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.convert_tokens_to_ids('[PAD]'), label_smoothing=0.1).to(device)

    #label_smoothing improve the accuracy of model

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)  # (B, seq_len)
            decoder_input = batch['decoder_input'].to(device)  # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)  # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)  # (B, 1, seq_len, seq_len), hide subsequent words besides padding

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask)  # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input,
                                          decoder_mask)  # (B, seq_len, d_model)
            proj_output = model.project(decoder_output)  # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device)  # (B, seq_len)

            # (Batch, seq_len, trg_vocab_size) --> (B* seq_len , trg_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_trg.vocab_size), label.view(-1))
            train_losses.append(loss.item())
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            #used by the tensorboard to keep track the loss
            global_step += 1

        run_validation(model, val_dataloader, tokenizer_src, tokenizer_trg, config['seq_len'], device,
                       lambda msg: batch_iterator.write(msg), global_step, writer)
        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

        avg_val_loss = run_validation(model, val_dataloader, tokenizer_src, tokenizer_trg, config['seq_len'], device,
                                  lambda msg: batch_iterator.write(msg), global_step, writer)
        val_losses.append(avg_val_loss)


    # At the end of train_model function
    #Visualize the training losses
    import matplotlib.pyplot as plt

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()




if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
