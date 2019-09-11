# -*- coding: utf-8 -*-
import numpy as np
import datetime
import time
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import csv
import json
import pandas as pd
import os
import random

class moneyflow_prediction():
    def __init__(self,using_old_data=False):
        self.gap_len = 62  # 跨月预测
        self.cut_points_num = 3  # div3次结果就是2^3=8份
        self.month_len = 31
        self.feature_len = 30
        self.test_len = 5*self.month_len
        self.need_len = 35
        self.cache_len = 4
        self.input_shape = (self.need_len, self.feature_len)
        self.normal=1e9

        self.outpredir='C:/Users/13486/Documents/a/aa/csv/shen'
        self.csvdir='C:/Users/13486/Documents/a/aa/csv/odpsCsv'
        self.holidayfile='C:/Users/13486/Documents/a/aa/csv/holiday/sholiday.csv'
        self.outtruedir='C:/Users/13486/Documents/a/aa/csv/act'
        self.algorithmdir='C:/Users/13486/Documents/a/aa/algorithm.tar/algorithm/algorithm'

        self.using_old_data=using_old_data
        self.epochs=20
        #self.epochs=100
        self.opt=Adam(lr=1e-3)

        self.adapt=1.1e9


    def create_data(self):
        if not self.using_old_data:
            datedict = {}
            moneylist = []
            rdatelist = ['min_arrive_date', 'max_arrive_date']  # 以后朝最小和最大到账时间的闭区间随机一天改进
           #将所有的csv文件拼接起来
            listcsv = os.listdir(self.csvdir)
            for filename in listcsv:
                if not '.csv' in filename:
                    continue
                print(filename)
                df_data = pd.read_csv(self.csvdir+'/'+filename)
                #对于dataframe df_data中的每一条数据，利用loc定位到第i条数据的date和money
                for i in range(df_data.shape[0]):
                    rdate = random.choice(rdatelist)
                    date = df_data.loc[i, rdate]
                    money = round(df_data.loc[i, 'rcv_amt'])

                    if money < 0:
                        continue
                    #append() 在list的尾部添加一个新的元素。
                    moneylist.append(money)
                    # https://blog.csdn.net/huyangshu87/article/details/52681478向字典中追加数据
                    if date in datedict:
                        datedict[date].append(money)
                    else:
                        datedict[date] = [money]

        #    print("db-1: type of date:", type(date))

            self.cutpoints = self.div_list(moneylist, 0, len(moneylist), self.cut_points_num)
            del moneylist


            df_holiday = pd.read_csv(self.holidayfile)
            dftemp = df_holiday[['date', 'isholiday']]
            self.df_holiday = dftemp.set_index('date')

            del df_holiday
            del dftemp


            train_dict = {}
            data_hist = np.zeros(shape=(2 ** self.cut_points_num), dtype=np.float)
            data_date = np.zeros(shape=(19), dtype=np.float)
            data_holiday = np.zeros(shape=(2), dtype=np.float)
            data_money = np.zeros(shape=(1), dtype=np.float)
            print('type of datedict_keys:',type(datedict.keys()))
            for date in datedict.keys():
                data_money[0] = sum(datedict[date]) / self.normal
                temphist = self.gethist(datedict[date])
                for i in range(data_hist.shape[0]):
                    data_hist[i] = temphist[i] / sum(datedict[date])
            
                date=str(date)
                d1 = datetime.datetime.strptime(str(date), "%Y%m%d")

                month_index = d1.month
                week_index = d1.weekday()
                data_date[6 + month_index] = data_money[0]
                data_date[week_index] = data_money[0]

                data_holiday[0] = float(self.df_holiday.loc[int(date),'isholiday'])

                train_dict[str(date)] = np.hstack([data_money, data_hist, data_date, data_holiday])


#            print("db-7: type of date:", type(date)) #str
 #           print("db-8: type of train_dict:", type(train_dict)) #

            date = '20120628'        #默认从这一天开始
            today = datetime.datetime.now()
            count = 1
            while date < str(today.year * 10000 + today.month * 100):
                if date in train_dict.keys():
                    train_dict[date][-1] = -1
                else:
                    train_dict[date] = [0] * self.feature_len
                    train_dict[date][-1] = -1
                    train_dict[date][-2] = 1

                if date[6:8] < '31':
                    date = str(int(date) + 1)
                elif date[4:6] < '12':
                    date = str(int(date[0:4]) * 100 + int(date[4:6]) + 1) + '01'
                else:
                    date = str(int(date[0:4]) + 1) + '0101'

                count += 1
                # print(date)

            temp_key = []
            for date in train_dict.keys():
                if date < '20120628':
                    temp_key.append(date)
            for date in temp_key:
                del train_dict[date]


            datedictkeys = sorted(train_dict.keys())
            for date_index in range(len(datedictkeys)):
                if date_index + self.gap_len < len(datedictkeys):
                    train_dict[datedictkeys[date_index]][-1] = train_dict[datedictkeys[date_index + self.gap_len]][-2]

                else:
                    date = datedictkeys[date_index]

                    #   下面的判断中默认了self.gap_len=62

                    if date[4:6] < '11':
                        date = str(int(date[0:4]) * 100 + int(date[4:6]) + 2) + date[6:8]
                    elif date[4:6] == '11':
                        date = str(int(date[0:4]) + 1) + '01' + date[6:8]
                    elif date[4:6] == '12':
                        date = str(int(date[0:4]) + 1) + '02' + date[6:8]

                    if int(date) in self.df_holiday.index.tolist():
                        train_dict[datedictkeys[date_index]][-1] = self.df_holiday.loc[int(date), 'isholiday'] + 0.0

                    else:
                        train_dict[datedictkeys[date_index]][-1] = 1.0

            #train_dict_tmp=train_dict
            for date in train_dict.keys():
                    train_dict[str(date)]=list(train_dict[str(date)])
                    #train_dict_tmp[str(date)]=list(train_dict[str(date)])
                    #print(date,train_dict_tmp[str(date)],type(train_dict_tmp[str(date)]))



            with open(self.algorithmdir+'/'+'data_x.json', 'w') as outfile:
                json.dump(train_dict, outfile, ensure_ascii=False)
                outfile.write('\n')
                #json.dump(train_dict_tmp, outfile, ensure_ascii=False)

            self.data_x = np.zeros(shape=(len(datedictkeys), self.feature_len), dtype=np.float)
            i = 0
            self.date=[]
            for date in datedictkeys:
                self.data_x[i] = train_dict[date]
                self.date = self.date + [date]
                
                #print(date)
                print(type(date))
                
                i += 1

            pd.DataFrame(self.data_x[-self.month_len:,0]*self.normal,columns=['true']).to_csv(self.outtruedir+'/'+datedictkeys[-self.month_len][:6]+'true.csv',index=False)



        else:
            with open(self.algorithmdir+'/'+"data_x.json", 'r') as load_f:
                train_dict = json.load(load_f)

            datedictkeys = sorted(train_dict.keys())
            self.data_x = np.zeros(shape=(len(datedictkeys), self.feature_len), dtype=np.float)
            self.date=[]
            i = 0
            for date in datedictkeys:
                self.data_x[i] = train_dict[date]
                
                self.date=self.date+[date]
                
                #print(date)
                i += 1

            # pd.DataFrame(self.data_x[-self.month_len:,0]*self.normal,columns=['true']).to_csv(self.outtruedir+'/'+datedictkeys[-self.month_len][:6]+'true.csv',index=False)


    def div_list(self,inputlist, start, end, k):
        if end - start < 2:
            return [inputlist[0]]
        sortlist = inputlist[start:end]
        totalsum = sum(sortlist) #将money list中的所有金额加和
        presum = 0
        i = 0
        while (i < len(sortlist) and presum < totalsum // 2):
            presum += sortlist[i]
            i += 1
        if i >= len(sortlist):
            i -= 1
        if k > 1:
            l1 = self.div_list(inputlist, start, start + i, k - 1)
            l2 = self.div_list(inputlist, start + i, end, k - 1)
            lout = l1 + [inputlist[start + i]] + l2 #将总金额分成8份，加到第几笔的时候到分位点
        else:
            lout = [inputlist[start + i]]
        return lout

    def gethist(self,inputlist):
        outhist = [0] * (2 ** self.cut_points_num)
        for x in inputlist:
            index = (2 ** self.cut_points_num) - 1
            while (index > 0 and x < self.cutpoints[index - 1]):
                index -= 1

            outhist[index] += x
        return outhist


    @staticmethod
    def drnn_model_large(input_shape):
        input1 = Input(shape=input_shape)  # 40days
        input2 = GaussianNoise(1e-3)(input1)
        gru1 = GRU(32, return_sequences=True)(input2)
        bn1 = BatchNormalization()(gru1)
        act1 = Activation('relu')(bn1)
        dropout1 = Dropout(0.2)(act1)
        gru2 = GRU(16, return_sequences=True)(dropout1)
        bn2 = BatchNormalization()(gru2)
        act2 = Activation('relu')(bn2)
        dropout2 = Dropout(0.2)(act2)
        gru3 = GRU(8, return_sequences=True)(dropout2)
        bn2 = BatchNormalization()(gru3)
        act3 = Activation('relu')(bn2)
        output = Dense(1)(act3)
        out_model = Model(input1, output)
        return out_model

    @staticmethod
    def drnn_model_medium(input_shape):
        input1 = Input(shape=input_shape)  # 40days
        input2 = GaussianNoise(1e-3)(input1)
        gru1 = GRU(8, return_sequences=True)(input2)
        bn1 = BatchNormalization()(gru1)
        act1 = Activation('relu')(bn1)
        dropout1 = Dropout(0.2)(act1)
        gru2 = GRU(8, return_sequences=True)(dropout1)
        bn2 = BatchNormalization()(gru2)
        act2 = Activation('relu')(bn2)
        dropout2 = Dropout(0.2)(act2)
        gru3 = GRU(4, return_sequences=True)(dropout2)
        bn2 = BatchNormalization()(gru3)
        act3 = Activation('relu')(bn2)
        output = Dense(1)(act3)
        out_model = Model(input1, output)
        return out_model


    @staticmethod
    def drnn_model_small(input_shape):
        input1 = Input(shape=input_shape)  # 40days
        input2 = GaussianNoise(1e-3)(input1)
        gru1 = GRU(32, return_sequences=True)(input2)
        bn1 = BatchNormalization()(gru1)
        act1 = Activation('relu')(bn1)
        dropout1 = Dropout(0.2)(act1)
        gru2 = GRU(16, return_sequences=True)(dropout1)
        bn2 = BatchNormalization()(gru2)
        act3 = Activation('relu')(bn2)
        output = Dense(1)(act3)
        out_model = Model(input1, output)
        return out_model


    def train(self):
        train_x = self.data_x[:-self.test_len]
        train_y = self.data_x[:-self.test_len, 0]
        test_x = self.data_x[-self.test_len - self.cache_len:]
        test_y = self.data_x[-self.test_len - self.cache_len:, 0]

        epochs = 20
        #epochs = 100
        batch_size = 4

        train_len = train_x.shape[0] - self.gap_len - self.need_len

        test_input = np.zeros(shape=(3, self.need_len, self.feature_len), dtype=np.float)
        test_output = np.zeros(shape=(3, self.need_len, 1), dtype=np.float)

        for i in range(3):
            test_input[i] = test_x[i * self.month_len:i * self.month_len + self.need_len]
            test_output[i, :, 0] = test_y[i * self.month_len + self.gap_len:i * self.month_len + self.gap_len + self.need_len]

        def get_train_batch():
            train_list = [x for x in range(train_len)]
            random.shuffle(train_list)
            train_list2 = [x for x in range(train_len // self.month_len)] * self.month_len
            random.shuffle(train_list2)
            train_batch_x = np.zeros(shape=(batch_size, self.need_len, self.feature_len), dtype=np.float)
            train_batch_y = np.zeros(shape=(batch_size, self.need_len, 1), dtype=np.float)
            i = 0
            k = 0
            while (i < (len(train_list) - batch_size)):
                for j in range(batch_size - 1):
                    train_batch_x[j] = train_x[train_list[i]:train_list[i] + self.need_len]
                    train_batch_y[j, :, 0] = train_y[train_list[i] + self.gap_len:train_list[i] + self.gap_len + self.need_len]
                    i += 1

                if k >= len(train_list2):
                    k = 0
                train_batch_x[-1] = train_x[train_list2[k]:train_list2[k] + self.need_len]
                train_batch_y[-1, :, 0] = train_y[train_list2[k] + self.gap_len:train_list2[k] + self.gap_len + self.need_len]
                k += 1
                yield train_batch_x, train_batch_y

        def cal_error1(y_pre, y_true):
            assert y_pre.shape == y_true.shape
            epsilon = 1e-3
            y_err = np.zeros(shape=(y_pre.shape[0], y_pre.shape[1]))
            for i in range(y_pre.shape[0]):
                for j in range(y_pre.shape[1]):
                    y_err[i, j] = abs((y_pre[i, j, 0] - y_true[i, j, 0]) / (y_true[i, j, 0] + epsilon))

            return np.mean(y_err, axis=0)

        def cal_error2(y_pre, y_true):
            assert y_pre.shape == y_true.shape
            y_err = np.zeros(shape=(y_pre.shape[0], y_pre.shape[1]))
            for i in range(y_pre.shape[0]):
                for j in range(y_pre.shape[1]):
                    y_err[i, j] = abs((y_pre[i, j, 0] - y_true[i, j, 0]))

            return np.mean(y_err, axis=0)


        my_model1 = self.drnn_model_large(self.input_shape)
        my_model1.compile(optimizer=self.opt, loss='mae')

        my_model2 = self.drnn_model_medium(self.input_shape)
        my_model2.compile(optimizer=self.opt, loss='mae')

        my_model3 = self.drnn_model_small(self.input_shape)
        my_model3.compile(optimizer=self.opt, loss='mae')

        best_loss1 = 1e8
        best_loss2 = 1e8
        best_loss3 = 1e8

        for i in range(self.epochs):
            mean_loss1 = 0
            mean_loss2 = 0
            mean_loss3 = 0
            for j, (batch_x, batch_y) in enumerate(get_train_batch()):
                mean_loss1 += my_model1.train_on_batch(batch_x, batch_y)
                mean_loss2 += my_model2.train_on_batch(batch_x, batch_y)
                mean_loss3 += my_model3.train_on_batch(batch_x, batch_y)

            mean_loss1 = mean_loss1 * 1e3 / (j + 1)
            mean_loss2 = mean_loss2 * 1e3 / (j + 1)
            mean_loss3 = mean_loss3 * 1e3 / (j + 1)

            pre_y1 = my_model1.predict(test_input)
            pre_y2 = my_model1.predict(test_input)
            pre_y3 = my_model1.predict(test_input)

            test_loss1 = my_model1.evaluate(test_input, test_output) * 1e3
            test_loss2 = my_model2.evaluate(test_input, test_output) * 1e3
            test_loss3 = my_model3.evaluate(test_input, test_output) * 1e3
            np.set_printoptions(precision=3, suppress=True)

            error1 = np.mean(cal_error2(pre_y1, test_output))
            error2 = np.mean(cal_error2(pre_y2, test_output))
            error3 = np.mean(cal_error2(pre_y3, test_output))
            error_mean=np.mean(cal_error2((pre_y3+pre_y2+pre_y1)/3, test_output))


            print("the train loss of [%d/%d] :  (large)%.3f   (medium)%.3f   (small)%.3f" % (i + 1, epochs, mean_loss1,mean_loss2,mean_loss3))
            print("the test  loss of [%d/%d] :  (large)%.3f   (medium)%.3f   (small)%.3f" % (i + 1, epochs, test_loss1,test_loss2,test_loss3 ))
            print("the test error of [%d/%d] :  (large)%.3f   (medium)%.3f   (small)%.3f" % (i + 1, epochs, error1,error2,error3))

            print("the mean test error of [%d/%d] :  %.3f" % (i + 1, epochs, error_mean))

            if (i > 5) and (test_loss1 < best_loss1):
                best_loss1 = test_loss1
                my_model1.save(self.algorithmdir+'/'+"weightlarge.h5")
                print('best weight1 save...')

            if (i > 5) and (test_loss2 < best_loss2):
                best_loss2 = test_loss2
                my_model2.save(self.algorithmdir+'/'+"weightmedium.h5")
                print('best weight2 save...')

            if (i > 5) and (test_loss3 < best_loss3):
                best_loss3 = test_loss3
                my_model3.save(self.algorithmdir+'/'+"weightsmall.h5")
                print('best weight3 save...')


    @staticmethod
    def cal_error(pre_y, true_y):
        assert pre_y.shape[0] == true_y.shape[0]
        epsilon = 1e-3
        mean_error = 0
        temp_pre = 0
        temp_true = 0
        error_1 = 0
        error_2 = 0
        error_3 = 0
        error_4 = 0
        for i in range(pre_y.shape[0]):
            temp_pre += pre_y[i]
            temp_true += true_y[i]
            temp_error = abs(pre_y[i] - true_y[i]) / (true_y[i] + epsilon)
            mean_error += temp_error

            if temp_error <= 0.1:
                error_1 += 1
            elif temp_error <= 0.2:
                error_2 += 1
            elif temp_error <= 0.4:
                error_3 += 1
            else:
                error_4 += 1

        sum_error = abs(temp_pre - temp_true) / temp_true

        print('drnn_mean_error:', mean_error / pre_y.shape[0])
        print('drnn_sum_error:', sum_error)
        print('error<=0.1:', error_1)
        print('error<=0.2:', error_2)
        print('error<=0.4:', error_3)
        print('error>0.4:', error_4)

    def predict(self):

        pre_x = self.data_x[-self.need_len:].reshape((1,self.need_len,self.feature_len))

        my_model1 = self.drnn_model_large(self.input_shape)
        my_model1.compile(optimizer=self.opt, loss='mae')

        my_model1.load_weights(self.algorithmdir+'/'+'weightlarge.h5')

        my_model2 = self.drnn_model_medium(self.input_shape)
        my_model2.compile(optimizer=self.opt, loss='mae')

        my_model2.load_weights(self.algorithmdir+'/'+'weightmedium.h5')

        my_model3 = self.drnn_model_small(self.input_shape)

        my_model3.compile(optimizer=self.opt, loss='mae')

        my_model3.load_weights(self.algorithmdir+'/'+'weightsmall.h5')

        pre_y1 = my_model1.predict(pre_x)[0,self.cache_len:,0] * self.adapt

        pre_y2 = my_model3.predict(pre_x)[0,self.cache_len:,0] * self.adapt

        pre_y3 = my_model3.predict(pre_x)[0,self.cache_len:,0] * self.adapt

        pre_y = (pre_y1 + pre_y2 + pre_y3) / 3

        pre_time=datetime.datetime.now()+datetime.timedelta(days=self.month_len)

        pd.DataFrame(pre_y, columns=['pre']).to_csv(
            self.outpredir + '/' + str(pre_time.year*100+pre_time.month) + 'pre.csv', index=False)

    def plot(self,index):
        if index>=self.data_x.size-self.gap_len-self.need_len:
            print('the input out of range')

        pre_x = self.data_x[index:index+self.need_len].reshape((1,self.need_len,self.feature_len))
        true_y = self.data_x[index+self.gap_len+self.cache_len:index+self.gap_len+self.need_len,0]*self.adapt

        my_model1 = self.drnn_model_large(self.input_shape)
        my_model1.compile(optimizer=self.opt, loss='mae')

        my_model1.load_weights(self.algorithmdir+'/'+'weightlarge.h5')

        my_model2 = self.drnn_model_medium(self.input_shape)
        my_model2.compile(optimizer=self.opt, loss='mae')

        my_model2.load_weights(self.algorithmdir+'/'+'weightmedium.h5')

        my_model3 = self.drnn_model_small(self.input_shape)

        my_model3.compile(optimizer=self.opt, loss='mae')

        my_model3.load_weights(self.algorithmdir+'/'+'weightsmall.h5')

        pre_y1 = my_model1.predict(pre_x)[0,self.cache_len:,0] * self.adapt

        pre_y2 = my_model3.predict(pre_x)[0,self.cache_len:,0] * self.adapt

        pre_y3 = my_model3.predict(pre_x)[0,self.cache_len:,0] * self.adapt


        print(self.date[66])

        print(self.date[self.gap_len+self.cache_len])
        pre_y=(pre_y1+pre_y2+pre_y3)/3

        self.cal_error(pre_y=pre_y,true_y=true_y)

        # import matplotlib.pyplot as plt
        # plt.figure(1)
        # plt.plot(pre_y, 'b')
        # plt.plot(true_y, 'r')
        # plt.xticks([x for x in range(self.need_len)],self.date[index+self.gap_len+self.cache_len:index+self.gap_len+self.need_len], rotation=90)
        # plt.grid()
        # plt.show()





if __name__=='__main__':


    print("start ----------------------------------------")
    my_model=moneyflow_prediction()
    print("create data----------------------------------------")
    my_model.create_data()
    print("train----------------------------------------")
    my_model.train()
    print("predict----------------------------------------")
    my_model.predict()
    print("end----------------------------------------")
   
    # my_model.plot(-159)








