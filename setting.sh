python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install sentencepiece
pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'