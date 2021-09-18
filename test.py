import os
import tensorflow as tf
import numpy as np
from read_utils import TextConverter, batch_generator
# from model import Model,Config
from model_att_bi_beamsearch import Model,Config

def main(_):
    converter = TextConverter(vocab_dir='data/vocabs', max_vocab=Config.vocab_size, seq_length = Config.seq_length)
    # 加载上一次保存的模型，测试是如果 True 则使用 beamsearch 预测，如果是 False 则使用 Greedy 预测
    model = Model(Config,'decode',True)
    model_path = os.path.join('models', Config.file_name)
    checkpoint_path = tf.train.latest_checkpoint(model_path)
    # 加载模型
    if checkpoint_path:
        model.load(checkpoint_path)

    while True:
        english_speek = input("上联:")
        english_speek = ' '.join(english_speek)
        english_speek = english_speek.split()
        en_arr, arr_len = converter.text_en_to_arr(english_speek)

        test_g = [np.array([en_arr,]), np.array([arr_len,])]
        output_ids = model.test(test_g, converter)
        strs = converter.arr_to_text(output_ids)
        print('下联:',strs)


if __name__ == '__main__':
    tf.app.run()