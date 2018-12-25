from enum import Enum


# 行为类型
class ActionType(Enum):
    BUY = 1  # 买入
    SELL = 2  # 卖出
    CONTINUE_BUY = 3  # 开始继续买入（目的是为了监控卖出效果的好坏，如果不这么做，卖出行为将只有短期反馈）


class BaseAction(object):
    def __init__(self, act_type, frame_index):
        self.act_type = act_type
        # 行为发生在哪一帧
        self.frame_index = frame_index
        # 行为评分
        self.reward = 0.


# 准备行为，仅仅用来选择使用哪个模型去做行为的决策
class PreAction(BaseAction):
    def __init__(self, act_type, frame_index, model_index):
        super(PreAction, self).__init__(act_type, frame_index)
        # 当前行为后，将使用哪个模型为后续行为做评估。
        self.model_index = model_index


# 真实行为
class RealAction(BaseAction):
    def __init__(self, act_type, frame_index, model_reward):
        super(RealAction, self).__init__(act_type, frame_index)
        # 记录一下模型算出来的奖励。如果奖励大于0，就按照模型的决策，否则就保持上个行为。注意：不在PRE_XX行为中使用这个
        # 这个值绝对值的大小就是之前PreAction的奖励分数（绝对值越大，说明模型评估得越清晰，这也是PreAction存在的意义：不同情况下使用不同模型做决策）
        self.model_reward = model_reward
