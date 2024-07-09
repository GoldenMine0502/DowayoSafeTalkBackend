from django.http import JsonResponse
from rest_framework.decorators import api_view

from mainapp.model.inference import DebertaInference

deberta_inference = DebertaInference()


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

    res = deberta_inference.inference(text)
    res1 = res[0]
    res2 = res[1]
    verify = res[2]

    return JsonResponse(
        {
            'percent1': res1,
            'percent2': res2,
            'verify': verify,
        }
    )
