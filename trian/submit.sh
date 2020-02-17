rm -f bce-python-sdk.zip
rm -rf bce-python-sdk
wget http://ai-studio-static.bj.bcebos.com/script/bce-python-sdk.zip
unzip -o bce-python-sdk.zip
cp -rf bce-python-sdk/* .

rm -f submit.py
wget http://ai-studio-static.bj.bcebos.com/script/submit.py
wget http://ai-studio-static.bj.bcebos.com/script/__init__.py
/usr/bin/python submit.py $1 $2