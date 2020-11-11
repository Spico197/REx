mkdir raw
cd raw
git clone https://github.com/SUDA-HLT/IPRE.git
cat IPRE/data/train/sent_train_1.txt IPRE/data/train/sent_train_2.txt > IPRE/data/train/sent_train.txt
cd ..
python convert.py
