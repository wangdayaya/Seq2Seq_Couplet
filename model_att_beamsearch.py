import os
import time
import tensorflow as tf

class Config(object):
    file_name = 'lstm_c_0918'  #保存模型文件
    embedding_dim = 128      # 词向量维度
    seq_length = 30        # 序列长度
    beam_size = 3
    vocab_size = 5000       # 词汇表达小
    num_layers= 2           # 隐藏层层数
    hidden_dim = 128        # 隐藏层神经元
    train_keep_prob = 0.8  # dropout保留比例
    learning_rate = 1e-3  # 学习率
    batch_size = 16  # 每批训练大小
    max_steps = 1000000  # 总迭代batch数
    log_every_n = 20  # 每多少轮输出一次结果
    save_every_n = 100  # 每多少轮校验模型并保存
    beam_search = 3

class Model(object):

    def __init__(self, config, mode, beam_search):
        self.config = config
        # 待输入的数据
        self.en_seqs = tf.placeholder(tf.int32, [None, self.config.seq_length], name='encode_input')
        self.en_length = tf.placeholder(tf.int32, [None], name='ec_length')
        self.zh_seqs = tf.placeholder(tf.int32, [None, self.config.seq_length], name='decode_input')
        self.zh_length = tf.placeholder(tf.int32, [None], name='zh_length')
        self.zh_seqs_label = tf.placeholder(tf.int32, [None, self.config.seq_length], name='label')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        # 两个全局变量
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.global_loss = tf.Variable(3, dtype=tf.float32, trainable=False, name="global_loss")
        self.mode = mode
        self.beam_search  = beam_search
        # seq2seq模型
        self.seq2seq()
        # 初始化session
        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def seq2seq(self):
        # 词嵌入层
        en_embedding = tf.get_variable('en_emb', [self.config.vocab_size, self.config.embedding_dim])
        zh_embedding = tf.get_variable('zh_emb', [self.config.vocab_size, self.config.embedding_dim])
        embedding_zero = tf.constant(0, dtype=tf.float32, shape=[1, self.config.embedding_dim])
        self.en_embedding = tf.concat([en_embedding, embedding_zero], axis=0)
        self.zh_embedding = tf.concat([zh_embedding, embedding_zero], axis=0)

        embed_en_seqs = tf.nn.embedding_lookup(self.en_embedding, self.en_seqs)
        embed_zh_seqs = tf.nn.embedding_lookup(self.zh_embedding, self.zh_seqs)

        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')

        # 在词嵌入上进行dropout
        embed_en_seqs = tf.nn.dropout(embed_en_seqs, keep_prob=self.keep_prob)
        embed_zh_seqs = tf.nn.dropout(embed_zh_seqs, keep_prob=self.keep_prob)

        def get_en_cell(hidden_dim):
            # 创建单个lstm
            enc_base_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim, forget_bias=1.0)
            return enc_base_cell

        with tf.variable_scope("encoder"):
            self.enc_cell = tf.nn.rnn_cell.MultiRNNCell([get_en_cell(self.config.hidden_dim) for _ in range(self.config.num_layers)])
            enc_output, enc_state = tf.nn.dynamic_rnn(cell=self.enc_cell, inputs=embed_en_seqs,  sequence_length=self.en_length, dtype=tf.float32)

        with tf.variable_scope("decoder_attention"):
            en_length = self.en_length
            # 使用 beamsearch 的时候需要将 encoder 的输出进行变换
            if self.beam_search:
                print("use beamsearch decoding..")
                enc_output = tf.contrib.seq2seq.tile_batch(enc_output, multiplier=self.config.beam_size)
                enc_state = tf.contrib.seq2seq.tile_batch(enc_state, self.config.beam_size)
                en_length = tf.contrib.seq2seq.tile_batch(self.en_length, multiplier=self.config.beam_size)
            # 定义 RNN 结构
            self.dec_cell = tf.nn.rnn_cell.MultiRNNCell([get_en_cell(self.config.hidden_dim) for _ in range(self.config.num_layers)])
            # 选择注意力权重计算模型，BahdanauAttention是使用一个隐藏层的前馈网络，memory_sequence_length是一个维度[batch_size]的张量，代表每个句子的长度
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(self.config.hidden_dim, enc_output, memory_sequence_length=en_length)
            # 将解码器和注意力模型一起封装成更高级的循环网络
            self.dec_cell = tf.contrib.seq2seq.AttentionWrapper(self.dec_cell, attention_mechanism, attention_layer_size = self.config.hidden_dim, name='Attention_Wrapper')
            # 使用 encoder 输出的 state 来初始化 deocoer 的起始状态
            decoder_initial_state = self.dec_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32).clone( cell_state=enc_state)
            # 全连接层，用来预测输出
            output_layer = tf.layers.Dense(self.config.vocab_size+1,  kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            # 训练时执行的代码
            if self.mode  == 'train':
                # 使用了 TrainingHelper 提升训练精度
                training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=embed_zh_seqs, sequence_length=self.zh_length, time_major=False,name='training_helper')
                training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=self.dec_cell, helper=training_helper, initial_state=decoder_initial_state, output_layer=output_layer)
                dec_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder, impute_finished=True,maximum_iterations=self.config.seq_length)
                with tf.name_scope("loss"):
                    self.decoder_logits_train = tf.identity(dec_output.rnn_output)
                    # self.decoder_predict_train = tf.argmax(self.decoder_logits_train, axis=-1,name='decoder_pred_train')
                    # 因为 decoder_logits_train 和要预测的标签长短不一，所以需要处理一下
                    current_ts = tf.to_int32(tf.minimum(tf.shape(self.zh_seqs)[1], tf.shape(self.decoder_logits_train)[1]))
                    target_sequence = tf.slice(self.zh_seqs_label, begin=[0, 0], size=[-1, current_ts])
                    mask_ = tf.sequence_mask(lengths=self.zh_length, maxlen=current_ts, dtype=tf.float32)
                    logits = tf.slice(self.decoder_logits_train, begin=[0, 0, 0], size=[-1, current_ts, -1])
                    self.mean_loss = tf.contrib.seq2seq.sequence_loss(logits=logits,targets=target_sequence, weights=mask_)
                with tf.name_scope("optimize"):
                    self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.mean_loss, global_step=self.global_step)
            # 预测解码时的代码
            elif self.mode == 'decode':
                start_tokens = tf.ones([self.batch_size, ], tf.int32) * 0
                end_token = 1
                #如果使用了 BeamSearchDecoder 预测
                if self.beam_search:
                    inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=self.dec_cell, embedding=self.zh_embedding,
                                                                             start_tokens=start_tokens, end_token=end_token,
                                                                             initial_state=self.dec_cell.zero_state(self.batch_size * self.config.beam_size,tf.float32).clone(cell_state=enc_state),
                                                                             beam_width=self.config.beam_size,
                                                                             output_layer=output_layer)
                # 如果使用的是 GreedyEmbeddingHelper 预测
                else:
                    decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.zh_embedding,  start_tokens=start_tokens, end_token=end_token)
                    inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=self.dec_cell, helper=decoding_helper, initial_state=decoder_initial_state,  output_layer=output_layer)
                # 预测输出
                decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder, maximum_iterations=self.config.seq_length)
                if self.beam_search:
                    # 对于使用beam_search的时候，它里面包含两项(predicted_ids, beam_search_decoder_output)
                    # predicted_ids: [batch_size, decoder_targets_length, beam_size],保存输出结果
                    # beam_search_decoder_output: BeamSearchDecoderOutput instance namedtuple(scores, predicted_ids, parent_ids)
                    self.decoder_predict_decode = decoder_outputs.predicted_ids[0][:,0]
                else:
                    # 对于不使用beam_search的时候，它里面包含两项(rnn_outputs, sample_id)
                    # rnn_output: [batch_size, decoder_targets_length, vocab_size]
                    # sample_id: [batch_size, decoder_targets_length], tf.int32
                    self.decoder_predict_decode = decoder_outputs.sample_id[0]

    def train(self, batch_train_g, model_path):
        with self.session as sess:
            for batch_en, batch_en_len, batch_zh, batch_zh_len, batch_zh_label in batch_train_g:
                start = time.time()
                feed = {self.en_seqs: batch_en,
                        self.en_length: batch_en_len,
                        self.zh_seqs: batch_zh,
                        self.zh_length: batch_zh_len,
                        self.zh_seqs_label: batch_zh_label,
                        self.keep_prob: self.config.train_keep_prob,
                        self.batch_size:self.config.batch_size}
                _, mean_loss = sess.run([self.optim, self.mean_loss ], feed_dict=feed)
                end = time.time()

                # control the print lines
                if self.global_step.eval() % self.config.log_every_n == 0:
                    print('step: {}/{}... '.format(self.global_step.eval(), self.config.max_steps),
                          'loss: {}... '.format(mean_loss),
                          '{:.4f} sec/batch'.format((end - start)))

                if (self.global_step.eval() % self.config.save_every_n == 0):
                    self.saver.save(sess, os.path.join(model_path, 'model'), global_step=self.global_step)
                if self.global_step.eval() >= self.config.max_steps:
                    break

    def test(self, test_g, zt):
        batch_en, batch_en_len = test_g
        output_ids = []
        feed = {self.en_seqs: batch_en,
                self.en_length: batch_en_len,
                self.keep_prob: 1.0,
                self.batch_size: len(batch_en)}
        decoder_predict_decode, = self.session.run([self.decoder_predict_decode,], feed_dict=feed)

        for i in decoder_predict_decode:
            char = zt.int_to_word(i)
            if char == '</s>':
                break
            output_ids.append(i)
        return output_ids


    def load(self, checkpoint):
        self.saver.restore(self.session, checkpoint)
        print('Restored from: {}'.format(checkpoint))

