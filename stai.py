import pandas as pd
import stframe as stf
import stai
import tensorflow as tf

EXPLORE = 2000000.  # 探索一段时间。探索期间，试错的概率会越来越小

FINAL_EPSILON = 0.0001  # 最后的试错概率
INITIAL_EPSILON = 0.0001  # 初始试错概率

DEPOSITE = 0.8  # 折扣率，折扣长期奖励

TRAIN_START_MONTH = 199909
TRAIN_END_MONTH = 201810
TEST_START_MONTH = 201511


if __name__ == '__main__':
    df_frame = pd.read_csv("data/frame.csv")
    df_offset = pd.read_csv("data/offset.csv")
    df_buy = pd.read_csv("data/buy.csv")
    df_sell = pd.read_csv("data/sell.csv")
    print(df_sell.columns.values.tolist())
    with tf.Session() as sess:
        frame_db = stai.FrameDB(df_frame, df_offset['offset'].values, df_buy, df_sell)
        frame_model = stai.TFFrameModel(3, 3, 3, df_sell.columns.values[1:], sess)
        frame_model.load()
        frame_trade = stai.FrameTrade(frame_model, frame_db)
        epsilon = INITIAL_EPSILON
        t = 0
        while True:
            # 产生一系列交易行为，并给他们打分
            trade_seq_list = frame_trade.create_trade_seq_list(epsilon=epsilon, start_month=TRAIN_START_MONTH, end_month=TRAIN_END_MONTH, max_buy_time_per_month=5, deposit=DEPOSITE)
            # 训练模型
            frame_trade.train(trade_seq_list)

            # scale down epsilon
            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            t += 1

            if t % 10000 == 0:
                frame_model.save()

            if t % 100 == 0:
                print("训练集买入卖出奖励(%.4f,%.4f)" % (frame_trade.cal_avg_buy_sell_reward(trade_seq_list)))
                trade_seq_list = frame_trade.create_trade_seq_list(epsilon=0, start_month=TEST_START_MONTH,
                                                               deposit=DEPOSITE)
                print("测试集买入卖出奖励(%.4f,%.4f)" % (frame_trade.cal_avg_buy_sell_reward(trade_seq_list)))
