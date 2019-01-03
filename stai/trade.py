import random
import stframe as stf
from .action import *
from .model import *


class FrameTrade(object):
    def __init__(self, frame_model, frame_db):
        self.frame_model = frame_model
        self.frame_db = frame_db

    # 创建一些交易行为，每个月最多进行max_buy_time_per_month次交易
    def create_trade_seq_list(self, epsilon, start_month=None, end_month=None, max_buy_time_per_month=None, deposit=0.8):
        trade_month = self.frame_db.trade_month
        trade_month = trade_month if start_month is None else trade_month[trade_month >= start_month]
        trade_month = trade_month if end_month is None else trade_month[trade_month <= end_month]
        trade_seq_list = []

        for month in trade_month:
            frame_list = self.frame_db.get_can_buy_frame(month)
            trade_list = self._create_ready_buy(epsilon, frame_list, max_buy_time_per_month)
            for trade in trade_list:
                trade_seq = [trade]
                self._create_trade_seq(epsilon, trade_seq)
                self.create_reward(trade_seq, deposit)
                if trade_seq[-1].act_type == ActionType.CONTINUE_BUY:
                    # 得到了一个完整的交易序列
                    trade_seq_list.append(trade_seq)

        return trade_seq_list

    def train(self, trade_seq_list):
        pre_data = {
            ActionType.BUY: FrameModelData(),
            ActionType.SELL: FrameModelData(),
            ActionType.CONTINUE_BUY: FrameModelData()
        }
        real_data = {
            ActionType.BUY: FrameModelData(),
            ActionType.SELL: FrameModelData(),
            ActionType.CONTINUE_BUY: FrameModelData()
        }
        for trade_seq in trade_seq_list:
            for id, act in enumerate(trade_seq):
                frame_feature = self.frame_db.get_feature(act.frame_index)
                model_index = act.model_index if id % 2 == 0 else trade_seq[id-1].model_index
                act_type = act.act_type
                reward = act.reward
                if id % 2 == 0:
                    pre_data[act_type].append(frame_feature, model_index, reward)
                else:
                    real_data[act_type].append(frame_feature, model_index, reward)

        self.frame_model.train_pre_buy_model(pre_data[ActionType.BUY])
        self.frame_model.train_pre_sell_model(pre_data[ActionType.SELL])
        self.frame_model.train_pre_continue_buy_model(pre_data[ActionType.CONTINUE_BUY])
        self.frame_model.train_buy_model(real_data[ActionType.BUY])
        self.frame_model.train_sell_model(real_data[ActionType.SELL])
        self.frame_model.train_continue_buy_model(real_data[ActionType.CONTINUE_BUY])

    @classmethod
    def cal_avg_buy_sell_reward(cls, trade_seq_list):
        cnt = 0
        sum = 0.0
        for trade_seq in trade_seq_list:
            for id, act in enumerate(trade_seq):
                if id % 2 == 1 and act.act_type != ActionType.CONTINUE_BUY:
                    sum += act.reward
                    cnt += 1
        return sum / cnt

    def create_reward(self, trade_seq, deposit=0.8):
        for i in range(len(trade_seq)-1, -1, -1):
            if i % 2 == 0:
                assert isinstance(trade_seq[i], PreAction)
                trade_seq[i].reward = abs(trade_seq[i+1].model_reward)
            else:
                assert isinstance(trade_seq[i], RealAction)
                if trade_seq[i].act_type == ActionType.CONTINUE_BUY:
                    if trade_seq[i].model_reward > 0:
                        # 再买入只需要看短期收益作为奖励
                        trade_seq[i].reward = self.frame_db.get_expect_return(trade_seq[i].frame_index)
                    else:
                        # 拒绝再买入也是一种短期指标，不用看得太远。需要看拒绝后的下跌作为远期奖励，再远就不看了
                        self._create_sell_reward(i, trade_seq, deposit)
                elif trade_seq[i].act_type == ActionType.SELL:
                    if i + 3 == len(trade_seq) and trade_seq[-1].act_type == ActionType.FORCE_STOP:
                        # 卖出发现早了
                        self._create_sell_reward(i, trade_seq, deposit)
                    elif trade_seq[i].model_reward > 0:
                        # 卖出时需要看到再买入的跌幅作为远期奖励，当前跌幅作为近期奖励（和拒绝再买入一样）
                        self._create_sell_reward(i, trade_seq, deposit)
                    else:
                        self._create_buy_reward(i, trade_seq, deposit)
                elif trade_seq[i].act_type == ActionType.BUY:
                    if i + 3 == len(trade_seq) and trade_seq[-1].act_type == ActionType.FORCE_STOP:
                        # 买早了
                        self._create_buy_reward(i, trade_seq, deposit)
                    elif trade_seq[i].model_reward > 0:
                        self._create_buy_reward(i, trade_seq, deposit)
                    else:
                        # 拒绝买入的奖励相当于卖出的奖励，通过后续的再买入行为去评估好坏
                        self._create_sell_reward(i, trade_seq, deposit)
                # else:
                    # 手动操作，不需要奖励去训练模型

    def _create_sell_reward(self, seq_index, trade_seq, deposit):
        start_frame_index = trade_seq[seq_index].frame_index
        if seq_index >= len(trade_seq):
            return
        end_frame_index = trade_seq[-1].frame_index
        far_reward = self.frame_db.get_delta_reward(end_frame_index, start_frame_index)
        near_reward = -self.frame_db.get_expect_return(start_frame_index)
        trade_seq[seq_index].reward = near_reward + far_reward * deposit

    def _create_buy_reward(self, seq_index, trade_seq, deposit):
        start_frame_index = trade_seq[seq_index].frame_index
        # 找到卖出序列或止损序列
        while True:
            end_seq_index = seq_index + 2
            if end_seq_index >= len(trade_seq):
                return
            if trade_seq[end_seq_index].act_type == ActionType.SELL and trade_seq[end_seq_index].model_reward > 0:
                break
            if trade_seq[end_seq_index].act_type == ActionType.FORCE_STOP:
                break
        end_frame_index = trade_seq[end_seq_index].frame_index
        far_reward = self.frame_db.get_delta_reward(start_frame_index, end_frame_index)
        near_reward = self.frame_db.get_expect_return(start_frame_index)
        trade_seq[seq_index].reward = near_reward + far_reward * deposit

    # 根据历史的交易序列，继续创建一个完整的交易序列
    def _create_trade_seq(self, epsilon, trade_seq):
        self._create_buy(epsilon, trade_seq)

    # 在frame_list中，选出max_buy_time个帧，准备进行交易
    def _create_ready_buy(self, epsilon, frame_list, max_buy_time):
        trade_list = []
        if max_buy_time is None:
            max_buy_time = len(frame_list)

        for _ in range(max_buy_time):
            if len(frame_list) == 0:
                break

            # 从frame_list中选择一帧，准备交易---------------------------------------------
            if random.random() < epsilon:
                # 随机选择一帧
                id = random.randint(0, len(frame_list-1))
                frame_index = frame_list.pop(id)
                model_index = random.randint(0, self.frame_model.pre_buy_model_num)
            else:
                best_frame_index = -1
                best_model_index = -1
                best_reward = -1000
                best_id = -1
                # todo 优化效率
                for id, frame_index in enumerate(frame_list):
                    model_index, reward = self.frame_model.get_best_pre_buy_model_index(self.frame_db.get_feature(frame_index))
                    if reward > best_reward:
                        best_reward = reward
                        best_frame_index = frame_index
                        best_model_index = model_index
                        best_id = id
                if best_reward < 0:
                    break
                frame_list.pop(best_id)
                frame_index = best_frame_index
                model_index = best_model_index

            trade_list.append(PreAction(ActionType.BUY, frame_index, model_index))
        return trade_list

    # 创建买入行为
    def _create_buy(self, epsilon, trade_seq):
        reward = self.frame_model.pred_buy_reward(self.frame_db.get_feature(trade_seq[-1].frame_index), trade_seq[-1].model_index)
        if random.random() < epsilon:
            # reward大于0决定真实买入，有一定概率会反着来，以便试错
            reward = -reward
        trade_seq.append(RealAction(ActionType.BUY, trade_seq[-1].frame_index, model_reward=reward))

        if reward > 0:
            # 预测可以买入
            self._create_pre_sell(epsilon, trade_seq)
        else:
            # 预测不可以买入，为了知道预测是否准确，还是需要产生一个continue buy行为，以便评估不买入决策的好坏
            self._create_pre_continue_buy(epsilon, trade_seq)

    # 创建准备卖出行为
    def _create_pre_sell(self, epsilon, trade_seq):
        frame_index = self.frame_db.get_next_frame_index(trade_seq[-1].frame_index, [stf.FrameType.top.value, stf.FrameType.top_delay.value, stf.FrameType.go_down.value, stf.FrameType.force_stop.value])
        if frame_index is None:
            return

        if self.frame_db.get_frame_type(frame_index) == stf.FrameType.force_stop.value:
            # 发现之前的买入行为错了，需要立即纠正-----------------------------------------------------
            trade_seq.append(PreAction(ActionType.FORCE_STOP, frame_index, 0))
            trade_seq.append(RealAction(ActionType.FORCE_STOP, frame_index, 0))
            return

        if random.random() < epsilon:
            model_index = random.randint(0, self.frame_model.pre_sell_model_num-1)
        else:
            model_index, _ = self.frame_model.get_best_pre_sell_model_index(self.frame_db.get_feature(frame_index))
        trade_seq.append(PreAction(ActionType.SELL, frame_index, model_index))
        self._create_sell(epsilon, trade_seq)

    # 创建继续买入行为（继续买入只是假装买入，不会真正买。目的是观察之前的卖出行为是否明智）
    def _create_pre_continue_buy(self, epsilon, trade_seq):
        frame_index = self.frame_db.get_next_frame_index(trade_seq[-1].frame_index, [stf.FrameType.bottom.value, stf.FrameType.bottom_delay.value, stf.FrameType.go_up.value, stf.FrameType.force_stop.value])
        if frame_index is None:
            return

        if self.frame_db.get_frame_type(frame_index) == stf.FrameType.force_stop.value:
            # 发现之前的行为错了，需要立即纠正-----------------------------------------------------------
            trade_seq.append(PreAction(ActionType.FORCE_STOP, frame_index, 0))
            trade_seq.append(RealAction(ActionType.FORCE_STOP, frame_index, 0))
            return

        if random.random() < epsilon:
            model_index = random.randint(0, self.frame_model.pre_buy_model_num-1)
        else:
            model_index, _ = self.frame_model.get_best_pre_continue_buy_model_index(self.frame_db.get_feature(frame_index))
        trade_seq.append(PreAction(ActionType.CONTINUE_BUY, frame_index, model_index))
        self._create_continue_buy(epsilon, trade_seq)

    # 创建卖出行为
    def _create_sell(self, epsilon, trade_seq):
        reward = self.frame_model.pred_sell_reward(self.frame_db.get_feature(trade_seq[-1].frame_index), trade_seq[-1].model_index)
        if random.random() < epsilon:
            # reward大于0决定真实买入，有一定概率会反着来，以便试错
            reward = -reward
        trade_seq.append(RealAction(ActionType.SELL, trade_seq[-1].frame_index, model_reward=reward))

        if reward > 0:
            # 预测可以卖出
            # 后续需要再假装买入以评估卖出效果
            self._create_pre_continue_buy(epsilon, trade_seq)
        else:
            # 预测持仓
            # 后续还需要继续卖出
            self._create_pre_sell(epsilon, trade_seq)

    def _create_continue_buy(self, epsilon, trade_seq):
        reward = self.frame_model.pred_continue_buy_reward(self.frame_db.get_feature(trade_seq[-1].frame_index), trade_seq[-1].model_index)
        if random.random() < epsilon:
            # reward大于0决定真实买入，有一定概率会反着来，以便试错
            reward = -reward
        trade_seq.append(RealAction(ActionType.CONTINUE_BUY, trade_seq[-1].frame_index, model_reward=reward))

