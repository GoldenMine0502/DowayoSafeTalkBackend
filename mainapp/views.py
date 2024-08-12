import json

import numpy as np
import torch
from django.http import JsonResponse, HttpResponse
from rest_framework.decorators import api_view

from DowayoSafeTalk.deberta_model import DebertaClassificationModel
from DowayoSafeTalk.preprocess import PreProcessKomoran
# from DowayoSafeTalk.preprocess import PreProcessKomoran
from DowayoSafeTalk.yamlload import Config

c = Config('DowayoSafeTalk/config/config.yml')
deberta_inference = DebertaClassificationModel(c, only_inference=True)
deberta_inference.load_weights(1)
process = PreProcessKomoran(use_space=False)
# checkpoint = torch.load(f'DowayoSafeTalk/deberta_{c.train.epoch}.pth', map_location=torch.device('cpu'))
# print(checkpoint)
# deberta_inference.model.load_state_dict(checkpoint['model_state_dict'])
# deberta_inference.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# Create your views here.
@api_view(['GET'])
def check(request):
    text = request.GET.get('text', None)

    if text is None:
        return JsonResponse(
            {
                'percent1': -1,
                'percent2': -1,
                'verify': 0,
            }
        )

    def remove_chosung(text):
        res = []

        for ch in text:
            if 'ㄱ' <= ch <= 'ㅎ':
                continue
            res.append(ch)

        return ''.join(res)

    texts = process.filter_text(text)
    text = ' '.join(texts)
    text = remove_chosung(text)

    print(text)
    logits, predicted_id = deberta_inference.inference(text)
    # print(logits.shape, predicted_id.shape)
    res1 = logits[0][0].detach().cpu().item()
    res2 = logits[0][1].detach().cpu().item()
    verify = predicted_id[0].detach().cpu().item()

    def softmax(X):
        exp_a = np.exp(X)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a

        return y

    softmaxs = softmax([res1, res2])
    softmax1 = round(softmaxs[0] * 100, 2)
    softmax2 = round(softmaxs[1] * 100, 2)

    response = {
        'percent1': softmax1,
        'percent2': softmax2,
        'verify': verify,
    }

    print(response)

    return HttpResponse(json.dumps(response), content_type="application/json")
