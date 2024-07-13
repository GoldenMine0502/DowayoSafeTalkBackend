from django.http import JsonResponse
from rest_framework.decorators import api_view

from DowayoSafeTalk.deberta_model import DebertaClassificationModel

c = Config('DowayoSafeTalk/config/config.yml')
deberta_inference = DebertaClassificationModel(c)


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
