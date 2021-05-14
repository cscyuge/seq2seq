from config import DCMN_Config
from run import build_seq2seq, build_dcmn
import torch
from dataset import seq_tokenize
from utils import decode_sentence, remove_unk, build_iterator
from bleu_eval_new import get_score
from tqdm import tqdm
import numpy as np
from dataset import build_dataset_eval
from dcmn import BertForMultipleChoiceWithMatch
import pickle

def main():
    config = DCMN_Config()
    eval_seq_dataset, eval_dcmn_dataset = build_dataset_eval(config)
    eval_dataloader = build_iterator(eval_seq_dataset, eval_dcmn_dataset, config)
    seq2seq, seq_optimizer, seq_scheduler, seq_loss_fun = build_seq2seq(config, 768, config.no_cuda)
    dcmn = BertForMultipleChoiceWithMatch.from_pretrained(config.bert_model, num_choices=config.num_choices)
    dcmn.to(config.dcmn_device)

    save_file_best = torch.load('./cache/best_save.data', map_location=torch.device('cuda:2'))
    dcmn.load_state_dict(save_file_best['dcmn_para'])
    seq2seq.load_state_dict(save_file_best['seq_para'])

    dcmn.eval()
    seq2seq.eval()

    results = []
    seq_srcs_all = []
    for step, (seq_batches, dcmn_batches) in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
        seq_srcs, seq_tars, cudics, k_cs = [[_[__] for _ in seq_batches] for __ in range(4)]
        outs = []

        if len(dcmn_batches) > 0:
            for p in range(0, len(dcmn_batches), config.batch_size):
                dcmn_batches_smaller = dcmn_batches[p: p + config.batch_size]
                input_ids, input_mask, segment_ids, doc_len, ques_len, option_len, labels = [
                    torch.LongTensor([_[__] for _ in dcmn_batches_smaller]).to(config.dcmn_device) for __ in range(7)]

                with torch.no_grad():
                    logits = dcmn(input_ids, segment_ids, input_mask, doc_len, ques_len, option_len)
                    outs_smaller = np.argmax(logits.detach().cpu().numpy(), axis=1)
                    outs.extend(outs_smaller)

        seq_srcs = remove_unk(seq_srcs, outs, k_cs)
        seq_srcs_all.extend(seq_srcs)
        src_ids, src_masks = seq_tokenize(seq_srcs, config)
        decoder_outputs, decoder_hidden, ret_dict = seq2seq([src_ids, src_masks], src_ids, 0.0, False)

        symbols = ret_dict['sequence']
        symbols = torch.cat(symbols, 1).data.cpu().numpy()
        results.extend(decode_sentence(symbols, config))
    with open('./outs/outs-new.pkl','wb') as f:
        pickle.dump(results, f)

    sentences = []
    for words in results:
        words = words.replace('[MASK] ', '')
        words = words.replace(' - ', '-').replace(' . ', '.').replace(' / ', '/')
        sentences.append(words.strip())

    with open('./result/tmp.out.txt', 'w', encoding='utf-8') as f:
        f.writelines([x.lower() + '\n' for x in sentences])
    bleu, hit, com, ascore = get_score()
    print('bleu:{}, hit:{}, com:{}, ascore:{}'.format(bleu, hit, com, ascore))


if __name__ == '__main__':
    main()