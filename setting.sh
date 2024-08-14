# pip3 freeze > requirements.txt
# pip install -r requirements.txt

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
#pip install sentencepiece
pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'

cp config/config-example.yml config/config.yml

# install apex
git clone https://github.com/NVIDIA/apex
cd apex || exit
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key...
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# otherwise
#pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./

cd ..



echo "install finished"