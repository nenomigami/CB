# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 18:41:47 2019

@author: ghdbs
"""

def pd_value_counts(pd):
    """
    feature을 돌면서 value_counts 를 만들고 dictionary 로 출력하는 함수
    input:
        pd - pandas.dataframe
    
    return:
        value_counts_table - (dictionary) table containing the counts of values
    """
    value_counts_table = {}
    for col in pd.columns:
        value_counts_table[col] = pd[col].value_counts()
    return value_counts_table    

def add_index_to_csv(file,index_name = "index"):
    import os
    import csv
    with open(file,'r') as f:
        with open("temp.csv",'w',newline = "") as d:
            reader = csv.reader(f)
            writer = csv.writer(d)
            header = next(reader)
            header.insert(0,index_name)
            writer.writerow(header)
            for i,row in enumerate(reader):
                row.insert(0,i)
                writer.writerow(row)
    os.remove(file)
    os.rename("temp.csv",file)

def load_eda_statistic_pd(filename, pd):
    import os
    import dill as pickle
    if filename in os.listdir():
         with open(filename,'rb') as f:
            return pickle.load(f)
    else:
        dim = pd.shape
        describe = pd.describe(include = "all")
        value_counts  = pd_value_counts(pd)
        isnull_table = pd.isnull().sum(axis=0)

        eda = {'dim' : dim, 'describe' : describe, 'value_counts' : value_counts, 'isnull_table' : isnull_table}
    
        with open(filename,"wb") as file:
            pickle.dump(eda,file)
        return eda

    
def reduce_mem_usage(df, verbose = True):
    import pandas as pd
    import numpy as np
    
    start_mem_usg = df.memory_usage().sum() / 1024**2
    
    print("Memory usage of properties dataframe is : {}MB".format(start_mem_usg))
    NAlist = []
    for col in df.columns:
        if df[col].dtype != object:
            if verbose == True:
                print("******************************")
                print("Column: ",col)
                print("dtype before: ",df[col].dtype)

            IsInt = False
            mx = df[col].max()
            mn = df[col].min()
            
            if not np.isfinite(df[col]).all():
                #모두 다 finite number가 아니면
                NAlist.append(col)
                df[col].fillna(mn-1,inplace=True)
                #minimum -1 로 대체한다
            
            # column이 정수로 바꿀 수 있는지 테스트
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.001 and result < 0.001:
                IsInt = True
            
            # 정수, unsigned 정수 타입으로 만들어주기
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)    
            #float은 32bit로 해둔다
            
            else:
                df[col] = df[col].astype(np.float32)
            
            # Print new column type
            if verbose == True:
                print("dtype after: ",df[col].dtype)
                print("******************************")

    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return df, NAlist

def plot_learning_curve(result, title = "", plot = True):
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib as plt
    df = pd.DataFrame(np.array([list(result[key].values())[0] for key in result]).T,
                      columns = ["learn","validation"])
    
    if plot == True:
        sns.lineplot(x = df.index,y = df["learn"],color = "green")
        sns.lineplot(x = df.index,y = df["validation"],color = "red")
        if title != "":
            plt.title(title)
    return df

if __name__ == "__main__":
    pass
