from django.http import JsonResponse
from rest_framework.decorators import api_view

from DowayoSafeTalk.deberta_model import DebertaClassificationModel
from DowayoSafeTalk.preprocess import PreProcessKomoran
from DowayoSafeTalk.yamlload import Config

c = Config('DowayoSafeTalk/config/config.yml')
deberta_inference = DebertaClassificationModel(c)
text_preprocess = PreProcessKomoran()

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

    text = text_preprocess.filter_text(text)

    logits, predicted_id = deberta_inference.inference(text)
    res1 = logits[0]
    res2 = logits[1]
    verify = predicted_id[0]

    return JsonResponse(
        {
            'percent1': res1,
            'percent2': res2,
            'verify': verify,
        }
    )
