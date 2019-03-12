from . import bpv, data, scoring
from .slimai import to_np

from kerosene import torch_util
import pandas as pd
import fire
import numpy as np
import pathlib
DATA_PATH=pathlib.Path('data')
def infer(model, dl):
    probs = []
    targs = []
    model.eval()
    for x, y in dl:
        probs.append(to_np(model(torch_util.variable(x))))
        targs.append(to_np(y))
    return np.concatenate(probs), np.concatenate(targs)


def predict(scores, tune_f1=False):
    if not tune_f1:
        return scores.argmax(axis=1)
    probs = scoring.softmax(scores)
    pcuts = scoring.pred_from_probs(probs)
    probs[probs < pcuts] = 0
    probs[:, -1] += 1e-9
    return probs.argmax(axis=1)

def ensemble_with_ir_system(filepath,category_encoder,totalscores,lambda_factor):#function for combining prediction results of our system and LSTM-BPV(s)
    irpredictionresults=pd.read_csv(
            filepath,
            sep='\t',
            header=None,
            names=('item','cat','prob'),
            )
            #print(total_targs.shape)
    predictedidx=category_encoder.encode(irpredictionresults.cat)
    predictedprobabilities=irpredictionresults.prob
    for pcatidx,prob,scores in zip(predictedidx,predictedprobabilities,totalscores):
           #scores[pcatidx]=(1-lambda_factor)*scores[pcatidx]+lambda_factor*prob
            #scores[pcatidx]=(1-lambda_factor)*scores[pcatidx]+lambda_factor*1
            scores[pcatidx]=(1-lambda_factor)*scores[pcatidx]+lambda_factor*(np.amax(scores)+0.1)

def ensemble_with_ir_system_revised(filepath,category_encoder,totalscores,lambda_factor):#function for combining prediction results of our system and LSTM-BPV(s)
    irpredictionresults=pd.read_csv(
            filepath,
            sep='\t',
            header=None,
            names=('item','cat','prob'),
            )
            #print(total_targs.shape)
    predictedidx=category_encoder.encode(irpredictionresults.cat)
    predictedprobabilities=irpredictionresults.prob
    for pcatidx,prob,scores in zip(predictedidx,predictedprobabilities,totalscores):
           #scores[pcatidx]=(1-lambda_factor)*scores[pcatidx]+lambda_factor*prob
            #scores[pcatidx]=(1-lambda_factor)*scores[pcatidx]+lambda_factor*1
            scores[pcatidx]=scores[pcatidx]+lambda_factor*(np.amax(scores)+0.1)
            #assert (scores.size==3008)
            #for y in range(pcatidx+1,3008):
             #   scores[y]=(1-lambda_factor)*scores[y]#+lambda_factor*0
                #i+=1

def main(forward=None, reverse=None, is_trainset=False, is_test=False, debug=False, i=0):
    n_emb, n_hid = 50, 512
    
    enc, cenc = data.load_encoders()
    n_inp, n_out = len(enc.itos), len(cenc.itos)

    models_by_dir = {
        False: forward.split(',') if forward else [],
        True: reverse.split(',') if reverse else [],
    }

    n_models = 0
    total_scores, total_targs = None, None
    for is_reverse, models in models_by_dir.items():
        if is_test:
            dl, revidx = data.load_test_dataloader(is_reverse)
        elif is_trainset:
            dl, _ = data.load_dataloaders(is_reverse)#,bs=32)
        else:
            _, dl = data.load_dataloaders(is_reverse)

        for model_name in models:
            model = data.load_model(
                torch_util.to_gpu(bpv.BalancedPoolLSTM(n_inp, n_emb, n_hid, n_out)),
                model_name,
            )

            scores, targs = infer(model, dl)
            if debug:
                preds = predict(scores)
                print(model_name, is_reverse, scoring.score(preds, targs))

            n_models += 1
            scores = scoring.logprob_scale(scores)
            if total_scores is None:
                total_scores, total_targs = scores, targs
            else:
                assert (targs == total_targs).all()
                total_scores += scores

    total_scores /= n_models
    for tune_f1 in False, True:
        if is_test:#infering on the Test Dataset
            ensemble_with_ir_system(DATA_PATH/'rdc-catalog-test-IB-SPL-DF-NormH1.tsv',cenc,total_scores,i)
            print("lambda="+str(i)+", ")
            pred = predict(total_scores, tune_f1=tune_f1)
            #print(data.save_test_pred(cenc, pred[revidx], tune_f1=tune_f1))
            #print(data.save_test_pred(cenc, pred, tune_f1=tune_f1))
            print(scoring.score(pred, total_targs))
        else:#infering on the validation dataset
            ensemble_with_ir_system(DATA_PATH/'predict-IB-winner-val-2019-01-11-2nd.tsv',cenc,total_scores,i)
            print("lambda="+str(i)+", ")
            print(scoring.score(predict(total_scores, tune_f1=tune_f1), total_targs))


if __name__ == '__main__':
    fire.Fire(main)
