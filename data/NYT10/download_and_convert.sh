mkdir raw
wget -P raw https://thunlp.oss-cn-qingdao.aliyuncs.com/opennre/benchmark/nyt10/nyt10_rel2id.json
wget -P raw https://thunlp.oss-cn-qingdao.aliyuncs.com/opennre/benchmark/nyt10/nyt10_train.txt
wget -P raw https://thunlp.oss-cn-qingdao.aliyuncs.com/opennre/benchmark/nyt10/nyt10_test.txt

python convert.py
