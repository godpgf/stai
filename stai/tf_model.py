from .model import *
import tensorflow as tf
import os

feature_num = 41
one_hot_size = 5


class TFFrameModel(FrameModel):
    def __init__(self, buy_model_num, sell_model_num, continue_buy_mode_num, columns, sess, lr=0.01):
        self.lr = lr
        self.optim = tf.train.GradientDescentOptimizer(self.lr, name='SGD')
        super(TFFrameModel, self).__init__(buy_model_num, sell_model_num, continue_buy_mode_num)
        self.columns = columns
        self.sess = sess
        self._create_input()
        self._create_variable()
        self._create_model()
        self.saver = tf.train.Saver()

    def _create_input(self):
        # 用来在不同情况下去不同模型
        self.model_id = tf.placeholder(dtype=tf.int32, shape=[None], name="model_id")

        self.input_list = [tf.placeholder(dtype=tf.int32, shape=[None], name="i_%d" % i) for i in range(feature_num)]
        self.input_onehot_list = [tf.one_hot(self.input_list[i], one_hot_size, 1.0, 0.0) for i in range(feature_num)]

        self.k_line = tf.expand_dims(tf.concat(self.input_onehot_list[:18], 1), 1)
        self.parting = tf.expand_dims(tf.concat(self.input_onehot_list[18:21], 1), 1)
        self.order = tf.expand_dims(tf.concat(self.input_onehot_list[21:30], 1), 1)
        self.sub_order = tf.expand_dims(tf.concat(self.input_onehot_list[30:32], 1), 1)
        self.segment = tf.expand_dims(tf.concat(self.input_onehot_list[32:35], 1), 1)
        self.gravitation = tf.expand_dims(tf.concat(self.input_onehot_list[35:37], 1), 1)
        self.first_trade = tf.expand_dims(tf.concat(self.input_onehot_list[37:41], 1), 1)

        self.label = tf.placeholder(dtype=tf.float32, shape=[None], name="label")

    def _create_variable(self):
        model_num = (self.buy_model_num + self.sell_model_num + self.continue_buy_model_num) * 2

        self.k_line_h1 = tf.Variable(tf.truncated_normal(shape=[model_num, 18 * one_hot_size, 32]), name="k_h1")
        self.k_line_gmf = tf.Variable(tf.truncated_normal(shape=[model_num, 32, 8]), name="k_gmf")
        self.k_line_mlp = tf.Variable(tf.truncated_normal(shape=[model_num, 32, 8]), name="k_mlp")

        self.parting_gmf = tf.Variable(tf.truncated_normal(shape=[model_num, 3 * one_hot_size, 8]), name="parting_gmf")
        self.parting_mlp = tf.Variable(tf.truncated_normal(shape=[model_num, 3 * one_hot_size, 8]), name="parting_mlp")

        self.order_h1 = tf.Variable(tf.truncated_normal(shape=[model_num, 9 * one_hot_size, 16]), name="order_h1")
        self.order_gmf = tf.Variable(tf.truncated_normal(shape=[model_num, 16, 8]), name="order_gmf")
        self.order_mlp = tf.Variable(tf.truncated_normal(shape=[model_num, 16, 8]), name="order_mlp")

        self.sub_order_gmf = tf.Variable(tf.truncated_normal(shape=[model_num, 2 * one_hot_size, 8]),
                                         name="sub_order_gmf")
        self.sub_order_mlp = tf.Variable(tf.truncated_normal(shape=[model_num, 2 * one_hot_size, 8]),
                                         name="sub_order_mlp")

        self.segment_gmf = tf.Variable(tf.truncated_normal(shape=[model_num, 3 * one_hot_size, 8]), name="segment_gmf")
        self.segment_mlp = tf.Variable(tf.truncated_normal(shape=[model_num, 3 * one_hot_size, 8]), name="segment_mlp")

        self.gravitation_gmf = tf.Variable(tf.truncated_normal(shape=[model_num, 2 * one_hot_size, 8]),
                                           name="gravitation_gmf")
        self.gravitation_mlp = tf.Variable(tf.truncated_normal(shape=[model_num, 2 * one_hot_size, 8]),
                                           name="gravitation_mlp")

        self.first_trade_gmf = tf.Variable(tf.truncated_normal(shape=[model_num, 4 * one_hot_size, 8]),
                                           name="first_trade_gmf")
        self.first_trade_mlp = tf.Variable(tf.truncated_normal(shape=[model_num, 4 * one_hot_size, 8]),
                                           name="first_trade_mlp")

        self.mlp_w1 = tf.Variable(tf.truncated_normal(shape=[model_num, 6 * 8, 32]), name="mlp_w1")
        self.mlp_w2 = tf.Variable(tf.truncated_normal(shape=[model_num, 32, 16]), name="mlp_w2")
        self.mlp_w3 = tf.Variable(tf.truncated_normal(shape=[model_num, 16, 8]), name="mlp_w2")

        self.w_out = tf.Variable(tf.truncated_normal(shape=[model_num, 16, 1]), name="w_out")
        self.b_out = tf.Variable(tf.truncated_normal(shape=[model_num, 1]), name="b_out")

    def _create_model(self):
        cur_k_line_h1 = tf.nn.relu(tf.matmul(self.k_line, tf.gather(self.k_line_h1, self.model_id)))
        cur_k_line_gmf = tf.matmul(cur_k_line_h1, tf.gather(self.k_line_gmf, self.model_id))
        cur_k_line_mlp = tf.nn.relu(tf.matmul(cur_k_line_h1, tf.gather(self.k_line_mlp, self.model_id)))

        cur_parting_gmf = tf.matmul(self.parting, tf.gather(self.parting_gmf, self.model_id))
        cur_parting_mlp = tf.nn.relu(tf.matmul(self.parting, tf.gather(self.parting_mlp, self.model_id)))

        cur_order_h1 = tf.nn.relu(tf.matmul(self.order, tf.gather(self.order_h1, self.model_id)))
        cur_order_gmf = tf.matmul(cur_order_h1, tf.gather(self.order_gmf, self.model_id))
        cur_order_mlp = tf.nn.relu(tf.matmul(cur_order_h1, tf.gather(self.order_mlp, self.model_id)))

        cur_segment_gmf = tf.matmul(self.segment, tf.gather(self.segment_gmf, self.model_id))
        cur_segment_mlp = tf.nn.relu(tf.matmul(self.segment, tf.gather(self.segment_mlp, self.model_id)))

        cur_gravitation_gmf = tf.matmul(self.gravitation, tf.gather(self.gravitation_gmf, self.model_id))
        cur_gravitation_mlp = tf.nn.relu(tf.matmul(self.gravitation, tf.gather(self.gravitation_mlp, self.model_id)))

        cur_first_trade_gmf = tf.matmul(self.first_trade, tf.gather(self.first_trade_gmf, self.model_id))
        cur_first_trade_mlp = tf.nn.relu(tf.matmul(self.first_trade, tf.gather(self.first_trade_mlp, self.model_id)))

        mlp_input = tf.concat(
            [cur_k_line_mlp, cur_parting_mlp, cur_order_mlp, cur_segment_mlp, cur_gravitation_mlp, cur_first_trade_mlp],
            2)
        mlp_layer_1 = tf.nn.relu(tf.matmul(mlp_input, tf.gather(self.mlp_w1, self.model_id)))
        mlp_layer_2 = tf.nn.relu(tf.matmul(mlp_layer_1, tf.gather(self.mlp_w2, self.model_id)))
        mlp_layer_3 = tf.nn.relu(tf.matmul(mlp_layer_2, tf.gather(self.mlp_w3, self.model_id)))

        def cross(feature):
            return tf.multiply(feature, cur_k_line_gmf) + tf.multiply(feature, cur_parting_gmf) + tf.multiply(feature,
                                                                                                              cur_order_gmf) + tf.multiply(
                feature, cur_segment_gmf) + tf.multiply(feature, cur_gravitation_gmf) + tf.multiply(feature,
                                                                                                    cur_first_trade_gmf)

        gmf = cross(cur_first_trade_gmf)
        concatenation = tf.concat([gmf, mlp_layer_3], 2)
        self.pred = tf.reshape(
            tf.matmul(concatenation, tf.gather(self.w_out, self.model_id)) + tf.gather(self.b_out, self.model_id), [-1])
        # todo 以后加正则项，目前不管他
        self.loss = tf.nn.l2_loss(self.pred - self.label)
        self.optimzer = self.optim.minimize(self.loss)

    def _get_fill_feature(self, frame_feature_list, model_id_list, reward_list=None):
        feed_dict = {self.input_list[id]: [] for id, column in enumerate(self.columns)}
        feed_dict[self.model_id] = []
        if reward_list is not None:
            feed_dict[self.label] = []
        for i in range(len(frame_feature_list)):
            frame_feature = frame_feature_list[i]
            for id, column in enumerate(self.columns):
                feed_dict[self.input_list[id]].append(frame_feature[column])
            feed_dict[self.model_id].append(model_id_list[i])
            if reward_list is not None:
                feed_dict[self.label].append(reward_list[i])
        return feed_dict

    def pred_pre_buy_reward(self, frame_feature, model_index):
        feed_dict = self._get_fill_feature([frame_feature], [model_index])
        return self.sess.run(self.pred, feed_dict=feed_dict)[0]

    def pred_buy_reward(self, frame_feature, model_index):
        feed_dict = self._get_fill_feature([frame_feature], [model_index + self.buy_model_num])
        return self.sess.run(self.pred, feed_dict=feed_dict)[0]

    def pred_pre_sell_reward(self, frame_feature, model_index):
        feed_dict = self._get_fill_feature([frame_feature], [model_index + self.buy_model_num * 2])
        return self.sess.run(self.pred, feed_dict=feed_dict)[0]

    def pred_sell_reward(self, frame_feature, model_index):
        feed_dict = self._get_fill_feature([frame_feature],
                                           [model_index + self.buy_model_num * 2 + self.sell_model_num])
        return self.sess.run(self.pred, feed_dict=feed_dict)[0]

    def pred_pre_continue_buy_reward(self, frame_feature, model_index):
        feed_dict = self._get_fill_feature([frame_feature],
                                           [model_index + self.buy_model_num * 2 + self.sell_model_num * 2])
        return self.sess.run(self.pred, feed_dict=feed_dict)[0]

    def pred_continue_buy_reward(self, frame_feature, model_index):
        feed_dict = self._get_fill_feature([frame_feature],
                                           [model_index + self.buy_model_num * 2 + self.sell_model_num * 2 + self.continue_buy_model_num])
        return self.sess.run(self.pred, feed_dict=feed_dict)[0]

    def _train_model(self, frame_model_data, model_id_offset):
        feature_list = []
        reward_list = []
        model_id_list = []
        for feature, model_id, reward in frame_model_data:
            feature_list.append(feature)
            reward_list.append(reward)
            model_id_list.append(model_id)
        feed_dict = self._get_fill_feature(feature_list, model_id_list, reward_list)
        return self.sess.run([self.loss, self.optimzer], feed_dict=feed_dict)

    def train_pre_buy_model(self, frame_model_data):
        return self._train_model(frame_model_data, 0)

    def train_pre_sell_model(self, frame_model_data):
        return self._train_model(frame_model_data, 2 *self.buy_model_num)

    def train_pre_continue_buy_model(self, frame_model_data):
        return self._train_model(frame_model_data, 2 * self.buy_model_num + 2 * self.sell_model_num)

    def train_buy_model(self, frame_model_data):
        return self._train_model(frame_model_data, self.buy_model_num)

    def train_sell_model(self, frame_model_data):
        return self._train_model(frame_model_data, 2 * self.buy_model_num + self.sell_model_num)

    def train_continue_buy_model(self, frame_model_data):
        return self._train_model(frame_model_data, 2 * self.buy_model_num + 2 * self.sell_model_num + self.continue_buy_model_num)

    def save(self, path="save/model.ckpt"):
        self.saver.save(self.sess, path)

    def load(self, path="save/model.ckpt"):
        fileName = path.split('/')[-1]
        if path and os.path.exists(path.replace(fileName, 'checkpoint')):
            self.saver.restore(self.sess, path)
        else:
            # init = tf.initialize_all_variables()
            init = tf.global_variables_initializer()
            self.sess.run(init)
