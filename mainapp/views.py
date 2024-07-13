import json

import torch
from django.http import JsonResponse, HttpResponse
from rest_framework.decorators import api_view

from DowayoSafeTalk.deberta_model import DebertaClassificationModel
# from DowayoSafeTalk.preprocess import PreProcessKomoran
from DowayoSafeTalk.yamlload import Config

c = Config('DowayoSafeTalk/config/config.yml')
deberta_inference = DebertaClassificationModel(c, only_inference=True)
checkpoint = torch.load(f'DowayoSafeTalk/deberta_{c.train.epoch}.pt', map_location=torch.device('cpu'))
# print(checkpoint)
deberta_inference.model.load_state_dict(checkpoint.state_dict())


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
    # text = text_preprocess.filter_text(text)
    # text = text.replace('/', '')

    print(text)
    logits, predicted_id = deberta_inference.inference(text)
    print(logits.shape, predicted_id.shape)
    res1 = round(logits[0][0].detach().cpu().item(), 2)
    res2 = round(logits[0][1].detach().cpu().item(), 2)
    verify = predicted_id.detach().cpu().item()

    response = {
        'percent1': res1,
        'percent2': res2,
        'verify': verify,
    }

    print(response)

    return HttpResponse(json.dumps(response), content_type="application/json")
