
from pykospacing import Spacing
from PyKomoran import *
from datasets import tqdm
from soynlp.normalizer import repeat_normalize



class PreProcessKomoran:
    def __init__(self):
        # print(__version__)
        self.komoran = Komoran("EXP")  # OR EXP
        self.datasets = [
            './dataset/data/korean_aihub1_result.txt',
            './dataset/data/korean_selectstar_result.txt',
            './dataset/data/korean_2runo_result.txt',
            './dataset/data/korean_smilegate_result.txt',
            './dataset/data/korean_kmhas_result.txt',
            './dataset/data/korean_kocohub_result.txt',
            './dataset/data/korean_womad_result.txt',
            './dataset/data/korean_zizun_result.txt',
            './dataset/data/korean_lgbt_result.txt',
        ]

        self.result_path = 'dataset/result.txt'
        # self.punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\\×™√²—–&'
        # self.punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2",
        #                  "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '”': '"', '“': '"', "£": "e",
        #                  '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-',
        #                  'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }

        self.spacing = Spacing()
        print('pykospacing test:', self.spacing('아 몬소리야 그건 또'))
    # def clean(self, text, punct, mapping):
    #     for p in mapping:
    #         text = text.replace(p, mapping[p])
    #
    #     for p in punct:
    #         text = text.replace(p, f' {p} ')
    #
    #     specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}
    #     for s in specials:
    #         text = text.replace(s, specials[s])
    #
    #     return text.strip()

    def clean_text(self, text):
        # pattern = '([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'  # E-mail제거
        # text = re.sub(pattern=pattern, repl='', string=text)
        # pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'  # URL제거
        # text = re.sub(pattern=pattern, repl='', string=text)
        # pattern = '([ㄱ-ㅎㅏ-ㅣ]+)'  # 한글 자음, 모음 제거
        # text = re.sub(pattern=pattern, repl='', string=text)
        # pattern = '<[^>]*>'  # HTML 태그 제거
        # text = re.sub(pattern=pattern, repl='', string=text)
        # pattern = '[^\s\n]'  # 특수기호제거
        # text = re.sub(pattern=pattern, repl='', string=text)
        # text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', string=text)
        # text = re.sub('\n', '.', string=text)

        text = text.replace('/', '')

        # print(text)
        text = self.spacing(text)
        text = repeat_normalize(text, num_repeats=2)


        # 즵 즺 즫 즥 즷 즴 즨 즹 즬 즿 즼 즽 즻 즻 즾
        text = (text.replace("#@이름#", "#")
                    .replace(" # ", "#")
                    .replace("##", "#")
                    .replace('즺', '짖')
                    .replace('즵', '집')
                    .replace('즫', '짇')
                    .replace('즥', '직')
                    .replace('즷', '짓')
                    .replace('즴', '짐')
                    .replace('즨', '진')
                    .replace('즹', '징')
                    .replace('즬', '질')
                    .replace('즿', '짛')
                    .replace('즼', '짘')
                    .replace('즽', '짙')
                    .replace('즻', '짗')
                    .replace('즾', '짚')
                    .replace('즤', '지')
                )

        return text

    def filter_text(self, text):
        def filter_sw(text):
            split = text.split("/")

            if len(split) == 1:
                return '#' not in text

            # return True
            if len(split) > 2:
                word = '/'.join(split[:-1])
                wtype = split[-1]
            else:
                word, wtype = split

            if wtype == 'NNP' or wtype == 'NNG':
                return True

            if wtype == 'VV' or wtype == 'VA':
                return True

            return False

        text = self.clean_text(text)

        res = self.komoran.get_plain_text(text).split(' ')

        res = list(filter(filter_sw, res))  # 필터링
        res = list(map(lambda x: x.split('/')[0], res))  # 대한민국/NNP 같은 단어가 있으면 슬래시 뒤 문자 떼버림

        return res

    def get_tokens(self):
        all_texts = []

        results = []

        for dataset in self.datasets:
            for line in open(dataset):
                all_texts.append(line.strip())

        for text in tqdm(all_texts):
            orig = text
            # print(text)
            texts = text.split("|")
            if len(texts) >= 3:
                first = ' '.join(texts[:-1])
                text = f'{first}|{texts[-1]}'

            text, label = text.split("|")
            res = self.filter_text(text)

            if res is None:
                continue

            results.append((res, label))

            # time.sleep(1)

        os = open(self.result_path, 'wt', encoding='utf-8')
        for all_text, label in tqdm(results):
            for i in range(len(all_text)):
                os.write(all_text[i])
                if i != len(all_text) - 1:
                    os.write('/')

            os.write('|')
            os.write(str(label))
            os.write('\n')

        os.close()

        # validation all texts are writed normally
        with open(self.result_path, 'rt', encoding='utf-8') as file:
            lines = len(file.readlines())
            print('lines:', lines)

            assert lines == len(results)


if __name__ == '__main__':
    # komoran = Komoran("EXP")  # OR EXP
    # print(komoran.get_plain_text("대한민국은  민주 공화  국이다.").split(' '))
    pre_komoran = PreProcessKomoran()
    pre_komoran.get_tokens()