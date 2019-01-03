import stframe as stf
import pandas as pd
import numpy as np


class FrameDB(object):
    def __init__(self, df_frame, offset, df_buy, df_sell):
        self.df_frame = df_frame
        self.offset = offset
        self.df_buy = df_buy
        self.df_sell = df_sell
        self.month_2_frame_index = self._get_month_2_frame_index(df_frame)
        self.month_2_buy_frame_index = {key: self._filter_buy_frame_index(value) for key, value in self.month_2_frame_index.items()}
        self.trade_month = np.unique(df_frame["date"].values // 100)

    # 得到某个月可以买入的所有帧
    def get_can_buy_frame(self, month):
        if month in self.month_2_buy_frame_index:
            return self.month_2_buy_frame_index[month]
        return None

    # 得到某帧的特征
    def get_feature(self, frame_index):
        if self._can_buy(frame_index):
            return self.df_buy.loc[frame_index]
        elif self._can_sell(frame_index):
            return self.df_sell.loc[frame_index]
        else:
            raise Exception("no no no!")
        return None

    def get_frame_type(self, frame_index):
        frame_type = int(self.df_frame['frame_type'].loc[frame_index])
        return frame_type

    def get_expect_return(self, frame_index):
        return self.df_frame['expect_return'].loc[frame_index]

    # 找到下一个关注的帧类型action_frame_type，直到超出范围
    def get_next_frame_index(self, frame_index, action_frame_type_list):
        action_frame_types = set(action_frame_type_list)
        max_frame_index = self.get_max_frame_index(frame_index)
        frame_index += 1
        while frame_index <= max_frame_index:
            frame_type = self.get_frame_type(frame_index)
            if frame_type in action_frame_types:
                break
            frame_index += 1
        return frame_index if frame_index <= max_frame_index else None

    # 得到当前帧所对应的最大帧
    def get_max_frame_index(self, frame_index):
        if frame_index <= self.offset[0]:
            return self.offset[0]
        left = 1
        right = len(self.offset) - 1
        while left < right:
            mid = ((left + right) >> 1)
            if self.offset[mid] >= frame_index and self.offset[mid-1] <= frame_index:
                return self.offset[mid]
            if self.offset[mid] > frame_index:
                right = mid - 1
            elif self.offset[mid] < frame_index:
                left = mid + 1
            else:
                raise Exception("no!")
        return self.offset[left]


    # 得到月份到frame_index映射
    @classmethod
    def _get_month_2_frame_index(cls, df_frame):
        month_2_frame_index = {}
        for frame_index, row in df_frame.iterrows():
            month = int(row["date"] // 100)
            if month not in month_2_frame_index:
                month_2_frame_index[month] = [int(frame_index)]
            else:
                month_2_frame_index[month].append(int(frame_index))
        return month_2_frame_index


    # 从一些帧中过滤出可以买入的
    def _filter_buy_frame_index(self, frame_index_list):
        df_frame = self.df_frame
        can_buy_frame_index_list = []
        for frame_index in frame_index_list:
            if pd.isna(df_frame['next_replace_frame'].loc[frame_index]) or int(df_frame['next_replace_frame'].loc[frame_index]) > frame_index:
                if self._can_buy(frame_index):
                    can_buy_frame_index_list.append(frame_index)
            return can_buy_frame_index_list

    def _can_buy(self, frame_index):
        # 这是一个有效的帧
        frame_type = self.get_frame_type(frame_index)
        if frame_type == stf.FrameType.bottom.value:
            return True
        if frame_type == stf.FrameType.bottom_delay.value:
            assert self.get_frame_type(int(self.df_frame['pre_frame'].loc[frame_index])) == stf.FrameType.bottom.value
            return True
        if frame_type == stf.FrameType.go_up.value:
            assert self.get_frame_type(int(self.df_frame['pre_frame'].loc[frame_index])) == stf.FrameType.bottom.value
            return True
        return False

    def _can_sell(self, frame_index):
        # 这是一个有效的帧
        frame_type = self.get_frame_type(frame_index)
        if frame_type == stf.FrameType.top.value:
            return True
        if frame_type == stf.FrameType.top_delay.value:
            assert self.get_frame_type(int(self.df_frame['pre_frame'].loc[frame_index])) == stf.FrameType.top.value
            return True
        if frame_type == stf.FrameType.go_down.value:
            assert self.get_frame_type(int(self.df_frame['pre_frame'].loc[frame_index])) == stf.FrameType.top.value
            return True
        return False


    # 得到两关键帧之间的价差
    def get_delta_reward(self, start_frame_index, end_frame_index):
        df_frame = self.df_frame
        start_close = df_frame["close"].loc[start_frame_index]
        start_atr = df_frame["atr"].loc[start_frame_index]
        end_close = df_frame["close"].loc[end_frame_index]
        end_atr = df_frame["atr"].loc[end_frame_index]
        return (end_close - start_close) / (start_atr + end_atr) * 2
