import heapq
import torch
import copy
from tqdm import tqdm
from helper import indexes2sent
import numpy as np
from metrics import Metrics
class BeamNode():
    def __init__(self, cur_idx, prob, decoded):
        self.cur_idx = cur_idx
        self.prob = prob
        self.decoded = decoded
        self.is_finished = False

    def __gt__(self, other):
        return self.prob > other.prob

    def __ge__(self, other):
        return self.prob >= other.prob

    def __lt__(self, other):
        return self.prob < other.prob

    def __le__(self, other):
        return self.prob <= other.prob

    def __eq__(self, other):
        return self.prob == other.prob

    def __ne__(self, other):
        return self.prob != other.prob

    def print_spec(self):
        print(f"ID: {self} || cur_idx: {self.cur_idx} || prob: {self.prob} || decoded: {self.decoded}")


class PriorityQueue():
    def __init__(self):
        self.queue = []

    def put(self, obj):
        heapq.heappush(self.queue, (obj.prob, obj))

    def get(self):
        return heapq.heappop(self.queue)[1]

    def qsize(self):
        return len(self.queue)

    def print_scores(self):
        scores = [t[0] for t in self.queue]
        print(scores)

    def print_objs(self):
        objs = [t[1] for t in self.queue]
        print(objs)

def evaluate_transformer(model, metrics, test_loader, vocab_desc, vocab_api, f_eval):
    ivocab_api = {v: k for k, v in vocab_api.items()}
    ivocab_desc = {v: k for k, v in vocab_desc.items()}
    device = next(model.parameters()).device
    recall_bleus, prec_bleus = [], []
    local_t = 0
    for old_descs, desc_lens, apiseqs, api_lens in tqdm(test_loader):
        # print("shape desc",old_descs.shape," shape api",apiseqs.shape)
        descs = torch.zeros_like(old_descs)
        descs[:, :-1] = old_descs[:, 1:50]
        descs[descs == 2] = 0
        # print("test No.",local_t)
        if local_t>1000:
            break

        desc_str = indexes2sent(descs[0].numpy(), vocab_desc)
        # print("test evaluate: desc_str",desc_str)
        src_data, desc_lens = [tensor.to(device) for tensor in [descs, desc_lens]]

        # print("source data", src_data)
        e_mask = (src_data != 0).unsqueeze(1).to(device)
        src_data = model.src_embedding(src_data)
        src_data = model.positional_encoder(src_data)
        e_output = model.encoder(src_data, e_mask)

        pred_trg = beam_search(model,device, e_output, e_mask)
        # print("predicted target", pred_trg)
        # print("actual target", apiseqs)
        pred_trg = np.array(pred_trg)
        # print("pred trg:",pred_trg)
        pred_sents, _ = indexes2sent( pred_trg, vocab_api)
        # print("pred sent:",pred_sents)
        pred_tokens = [sent.split(' ') for sent in pred_sents]
        ref_str, _ = indexes2sent(apiseqs[0].numpy(), vocab_api)
        ref_tokens = ref_str.split(' ')
        # print("pred token:",pred_tokens)
        # print("actu token:",ref_tokens)
        max_bleu, avg_bleu = metrics.sim_bleu(pred_tokens, ref_tokens)
        recall_bleus.append(max_bleu)
        prec_bleus.append(avg_bleu)
        local_t += 1
        f_eval.write("Batch %d \n" % (local_t))# print the context        
        f_eval.write(f"Query: {desc_str} \n")
        f_eval.write("Target >> %s\n" % (ref_str.replace(" ' ", "'")))# print the true outputs 

    recall_bleu = float(np.mean(recall_bleus))
    prec_bleu = float(np.mean(prec_bleus))
    f1 = 2 * (prec_bleu * recall_bleu) / (prec_bleu + recall_bleu + 10e-12)

    report = "Avg recall BLEU %f, avg precision BLEU %f, F1 %f" % (recall_bleu, prec_bleu, f1)
    print(report)
    f_eval.write(report + "\n")
    print("Done testing")


# def inference_transformer(model, src_data):
#     device = next(model.parameters()).device
#     # default as beam search
#     e_mask = (src_data != 0).unsqueeze(1).to(device)
#     src_data = model.src_embedding(src_data)
#     src_data = model.positional_encoder(src_data)
#     e_output = model.encoder(src_data, e_mask)
#     result = beam_search(model, e_output, e_mask)
#     return result


def beam_search(model, device, e_output, e_mask):
    print("beam searching")
    cur_queue = PriorityQueue()
    for k in range(8):
        cur_queue.put(BeamNode(1, -0.0, [1]))

    finished_count = 0

    for pos in range(50):
        new_queue = PriorityQueue()
        for k in range(8):
            node = cur_queue.get()
            if node.is_finished:
                new_queue.put(node)
            else:
                trg_input = torch.LongTensor(node.decoded + [0] * (50 - len(node.decoded))).to(device)  # (L)
                d_mask = (trg_input.unsqueeze(0) != 0).unsqueeze(1).to(device)  # (1, 1, L)
                nopeak_mask = torch.ones([1, 50, 50], dtype=torch.bool).to(device)
                nopeak_mask = torch.tril(nopeak_mask)  # (1, L, L) to triangular shape
                d_mask = d_mask & nopeak_mask  # (1, L, L) padding false

                trg_embedded = model.trg_embedding(trg_input.unsqueeze(0))
                trg_positional_encoded = model.positional_encoder(trg_embedded)
                decoder_output = model.decoder(
                    trg_positional_encoded,
                    e_output,
                    e_mask,
                    d_mask
                )  # (1, L, d_model)

                output = model.softmax(
                    model.output_linear(decoder_output)
                )  # (1, L, trg_vocab_size)

                output = torch.topk(output[0][pos], dim=-1, k=8)
                last_word_ids = output.indices.tolist()  # (k)
                last_word_prob = output.values.tolist()  # (k)

                for i, idx in enumerate(last_word_ids):
                    new_node = BeamNode(idx, -(-node.prob + last_word_prob[i]), node.decoded + [idx])
                    if idx == 2:
                        new_node.prob = new_node.prob / float(len(new_node.decoded))
                        new_node.is_finished = True
                        finished_count += 1
                    new_queue.put(new_node)

        cur_queue = copy.deepcopy(new_queue)

        if finished_count == 8:
            break

    decoded_output = cur_queue.get().decoded

    if decoded_output[-1] == 2:
        decoded_output = decoded_output[1:-1]
    else:
        decoded_output = decoded_output[1:]

    # print("decoded output", decoded_output)
    # print("decoded output len", len(decoded_output))

    return [decoded_output]
