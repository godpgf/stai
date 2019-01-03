class FrameModelData(object):
    def __init__(self):
        self.frame_feature_list = []
        self.model_index_list = []
        self.reward_list = []

    def append(self, frame_feature, model_index, reward):
        self.frame_feature_list.append(frame_feature)
        self.model_index_list.append(model_index)
        self.reward_list.append(reward)


class FrameModel(object):
    def __init__(self, buy_model_num, sell_model_num, continue_buy_model_num):
        self.buy_model_num = buy_model_num
        self.sell_model_num = sell_model_num
        self.continue_buy_model_num = continue_buy_model_num

    # 找到最适合frame_index的预购买模型(frame_feature中包含了买入行为发生在底分型停顿还是后续的追涨)
    def get_best_pre_buy_model_index(self, frame_feature):
        return self._get_best_pre_model(frame_feature, self.pred_pre_buy_reward)

    # 找到最适合frame_index的预卖出模型(frame_feature中包含了卖出行为发生在顶分型停顿还是后续的杀跌)
    def get_best_pre_sell_model_index(self, frame_feature):
        return self._get_best_pre_model(frame_feature, self.pred_pre_sell_reward)

    def get_best_pre_continue_buy_model_index(self, frame_feature):
        return self._get_best_pre_model(frame_feature, self.pred_continue_buy_reward)

    def _get_best_pre_model(self, frame_feature, pre_fun):
        best_reward = -1000
        best_model_index = -1
        for i in range(self.sell_model_num):
            reward = pre_fun(frame_feature, i)
            if reward > best_reward:
                best_reward = reward
                best_model_index = i
        return best_model_index, best_reward

    # 需要重载-----------------------------------------------------------------------------
    def pred_pre_buy_reward(self, frame_feature, model_index):
        return 0

    def pred_buy_reward(self, frame_feature, model_index):
        return 0

    def pred_pre_sell_reward(self, frame_feature, model_index):
        return 0

    def pred_sell_reward(self, frame_feature, model_index):
        return 0

    def pred_pre_continue_buy_reward(self, frame_feature, model_index):
        return 0

    def pred_continue_buy_reward(self, frame_feature, model_index):
        return 0

    def train_pre_buy_model(self, frame_model_data):
        pass

    def train_pre_sell_model(self, frame_model_data):
        pass

    def train_pre_continue_buy_model(self, frame_model_data):
        pass

    def train_buy_model(self, frame_model_data):
        pass

    def train_sell_model(self, frame_model_data):
        pass

    def train_continue_buy_model(self, frame_model_data):
        pass

    def save(self, path="save/model.ckpt"):
        pass

    def load(self, path="save/model.ckpt"):
        pass