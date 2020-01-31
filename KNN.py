#coding:utf-8

import numpy as np
import operator   #内置的算数比较包
import xlrd    #用于操作表格

##给出训练数据以及对应的类别
def createDataSet():
    group = np.array([[1.0,2.0],[1.2,0.1],[0.1,1.4],[0.3,3.5]])
    labels = ['A','A','B','B']
    return group,labels

###通过KNN进行分类
def classify(input,dataSet,label,k):
    dataSize = dataSet.shape[0]
    ####计算欧式距离
    diff = np.tile(input,(dataSize,1)) - dataSet
    sqdiff = diff ** 2
    squareDist = np.sum(sqdiff,axis = 1)###行向量分别相加，从而得到新的一个行向量
    dist = squareDist ** 0.5
    
    ##对距离进行排序
    sortedDistIndex = np.argsort(dist)##argsort()根据元素的值从大到小对元素进行排序，返回下标

    classCount={}
    for i in range(k):
        voteLabel = label[sortedDistIndex[i]]
        ###对选取的K个样本所属的类别个数进行统计
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
    ###选取出现的类别次数最多的类别
    maxCount = 0
    for key,value in classCount.items():
        if value > maxCount:
            maxCount = value
            classes = key

    return classes

if __name__ == '__main__':
    DataSet = xlrd.open_workbook(r'./DataSet/KNN.xls')
    sheet = DataSet.sheet_by_index(0)
    print("数据共：",sheet.ncols," 列",sheet.nrows," 行")
    group = np.array([[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],dtype=np.float)
    labels = ['?','?','?','?','?','?']
    print(group.shape[0])
    for i in range(sheet.nrows):
        group[i] = sheet.cell_value(i,0),sheet.cell_value(i,1)
        labels[i] = sheet.cell_value(i,2)
        print("第 ",i," 行的数据为",group[i],"类别为",labels[i])
    inputdata = eval(input("please input your data ( separated by commas):"))
    K = eval(input("input K value ：")) #eval 将输入时自动添加的""去掉
    DataSet = np.array(inputdata)
    #print(type(inputdata))
    #print(labels)
    #print(type(indata),np.shape(indata),indata)
    output = classify(DataSet,group,labels,K)
    print("resout of the class is : ",output)
        