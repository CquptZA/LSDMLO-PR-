#!/usr/bin/env python
# coding: utf-8
from NeedFunction import *
# In[ ]:
def Distance(df,weights,k,p1):
    n = df.shape[0]  
    distances = np.zeros((n, n))  
    data=np.array(df)
    if p1==np.inf:
        for i in range(n):
            for j in range(n):
                distances[i, j]=np.max(weights* np.abs(data[i] - data[j]))               
    else:
        for i in range(n):
            for j in range(i, n): 
                distances[i, j] = np.power(np.sum(weights * np.abs(data[i] - data[j])**p1), 1/p1)
        distances = distances + distances.T - np.diag(distances.diagonal())
    sorted_indices = np.argsort(distances, axis=1)
    sorted_indices = sorted_indices[:, ::1]
    topk_indices= sorted_indices[:, :k]
    return topk_indices
def assign_weights(arr):
    arr =np.abs(arr)
    weights = np.exp(arr - np.max(arr)) / np.sum(np.exp(arr - np.max(arr)))
    return weights
def label_assign_distance(v1, v2, w):
    distance = np.power(np.sum(w * np.abs(v1 - v2)**2), 1/2)
    return distance
def label_similarity(Y,k):
    m = Y.shape[1]
    cos_sim = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i == j:
                cos_sim[i, j] = 1.0
            else:
                cos_sim[i, j] = np.dot(Y[:, i], Y[:, j]) / (np.linalg.norm(Y[:, i]) * np.linalg.norm(Y[:, j]))
    upper_triangle_indices = np.triu_indices(cos_sim.shape[1], k=1)
    upper_triangle_values = cos_sim [upper_triangle_indices]
    top_k_indices = np.argpartition(upper_triangle_values, -k)[-k:]
    result_matrix = np.zeros_like(cos_sim)
    result_matrix[upper_triangle_indices[0][top_k_indices], upper_triangle_indices[1][top_k_indices]] = upper_triangle_values[top_k_indices]
    return result_matrix
def Labeltype(X,y,label_names):
    ImbalanceRatioMatrix,MeanIR,_=Imbalance(X,y)
    DifferenceImbalanceRatioMatrix=[i-MeanIR for i in ImbalanceRatioMatrix]
    MinLabelIndex=[]
    MajLabelIndex=[]
    count=0
    for i in (DifferenceImbalanceRatioMatrix):
        if i>0:
            MinLabelIndex.append(count)
        else:
            MajLabelIndex.append(count)
        count+=1
    MinLabelName=[]
    MajLabelName=[]
    for i in MinLabelIndex:
        MinLabelName.append(label_names[i][0])
    for i in MajLabelIndex:
        MajLabelName.append(label_names[i][0])
    MinLabeldic=dict(zip(MinLabelIndex,MinLabelName))
    MajLabeldic=dict(zip(MajLabelIndex,MajLabelName))
    return MinLabeldic,MajLabeldic
def LSDMLOsampling(df1, df2, W, sp,feature_names,label_names):
    ImrMatrix=ImR(np.array(df1),np.array(df2))
    n_neighbors = 5
    p=2
    cos_sim = label_similarity(np.array(df2),10)
    non_zero_indices = [np.where(row != 0)[0].tolist() for row in cos_sim]
    row_sums = cos_sim.sum(axis=1)
    normalized_label_weight = cos_sim / row_sums[:, np.newaxis]
    MinLabeldic, MajLabeldic = Labeltype(np.array(df1), np.array(df2),label_names)
    ImbalanceRatioMatrix, MeanIR, countmatrix = Imbalance(np.array(df1), np.array(df2))
    MinLabelindex = list(MinLabeldic.keys())
    C = np.zeros((df1.shape[0], len(MinLabelindex)))
    C_hat=np.zeros((df1.shape[0], len(MinLabelindex))) 
    indices_dict = {}
    for tail_label in MinLabelindex:
        all_relevant=non_zero_indices[tail_label]
        sub_index = np.where(df2[MinLabeldic[tail_label]] == 1)[0]
        idx = MinLabelindex.index(tail_label)      
        W_tail_label = W[:, tail_label]
        sorted_indices = np.argsort(W_tail_label)
        sorted_column = W_tail_label[sorted_indices]
        featureWeight = assign_weights(W_tail_label)
        indices = Distance(df1, featureWeight, n_neighbors + 1, p)
        indices_dict[tail_label] = indices
        for i in range(df1.shape[0]):
            if df2.iloc[i,tail_label]==0:
                continue
            count=0
            for j in indices[i, 1:]:
                if df2.iloc[i,tail_label]==df2.iloc[j,tail_label]:
                    count +=1
            C[i, MinLabelindex.index(tail_label)] = count / n_neighbors
            count1list=[]
            if all_relevant:   
                for k in all_relevant:
                    count1=0
                    for j in indices[i, :]:
                        if df2.iloc[j, k] == 1:
                            count1+= 1
                    count1list.append(count1)
                C_hat[i, MinLabelindex.index(tail_label)]=max(count1list) / n_neighbors      
    Ins_Weight=np.zeros(df1.shape[0])  
    tem = np.zeros([df1.shape[0], len(MinLabelindex)])
    tem_hat = np.zeros([df1.shape[0], len(MinLabelindex)])
    for j in range(len(MinLabelindex)):
        SumC = 0.0
        sum_C_1 = 0.0
        c = 0
        c_1 = 0

        for i in range(df1.shape[0]):
            if C[i, j] < 1 and C[i, j] != 0:
                SumC += C[i, j]
                c += 1
            if C_hat[i, j] != 0:
                sum_C_1 += C_hat[i, j]
                c_1 += 1

            if SumC != 0.0 and c != 0:
                if C[i, j] < 1 and C[i, j] != 0:
                    tem[i, j] = C[i, j] / SumC
            else:
                tem[i, j] = 0

            if sum_C_1 != 0.0 and c_1 != 0:
                if C_hat[i, j] != 0:
                    tem_hat[i, j] = C_hat[i, j] / sum_C_1
            else:
                tem_hat[i, j] = 0
    SumW = 0
    for i in range(df1.shape[0]):
        for j in range(len(MinLabelindex)):
            if tem[i, j] != 0:
                Ins_Weight[i] += tem[i, j] + tem_hat[i, j]
        SumW += Ins_Weight[i]
    non_zero_elements = []

    for row in tem:
        for element in row:
            if element != 0:
                non_zero_elements.append(element)      
    n_sample = int(df1.shape[0] * sp)
    new_X = np.zeros((n_sample, df1.shape[1]))
    target = np.zeros((n_sample, df2.shape[1]))
    count = 0     
    while count < n_sample:
        random_count=np.random.random()*SumW
        seed=0
        s=0
        for k in range(len(Ins_Weight)):
            s+=Ins_Weight[k]
            if(random_count<=s):
                seed=k
                break    
        seedtype = np.where(np.array(df2.iloc[seed]) == 1)[0]
        set1 = set(seedtype)
        set2 = set(MinLabelindex)
        intersection = set1.intersection(set2)
        intersection_list = list(intersection)
        if not intersection_list:
            continue  # 当 intersection_list 为空时，跳过当前循环
        select_index=np.random.choice(intersection_list)
        reference = np.random.choice(indices_dict[select_index][seed, 1:])
        all_point = indices_dict[select_index][seed, :]
        nn_df = df2[df2.index.isin(all_point)]
        ser = nn_df.sum(axis = 0, skipna = True)   
        for j in range(df1.shape[1]):
            ratio = np.random.random()
            if feature_names[j][1] == 'NUMERIC':
                new_X[count, j] = df1.iloc[seed, j] + ratio * (df1.iloc[reference, j] - df1.iloc[seed, j])
            elif feature_names[j][1] == ['YES', 'NO'] or feature_names[j][1] == ['0', '1']:
                rmd = np.random.choice([True, False])
                if rmd:
                    new_X[count, j] = df1.iloc[seed, j]
                else:
                    new_X[count, j] = df1.iloc[reference, j]
            else:
                new_X[count, j] = df1.iloc[seed, j]    
        for j in range(df2.shape[1]):
            if df2.iloc[seed, j] == df2.iloc[reference, j]:
                target[count, j]=df2.iloc[seed, j]
            else:
                featureWeight = assign_weights(W[:, j])
                distance1 = label_assign_distance(np.array(df1.iloc[seed, :]), new_X[count, :], featureWeight)
                distance2 = label_assign_distance(np.array(df1.iloc[reference, :]), new_X[count, :], featureWeight)
                if distance1 <= distance2:
                    target[count, j] = df2.iloc[seed, j]
                else:
                    target[count, j] = df2.iloc[reference, j]
        count += 1
    new_X = pd.DataFrame(new_X, columns=[x[0] for x in feature_names])
    target = pd.DataFrame(target, columns=[y[0] for y in label_names])
    LSDMLO_new_X = pd.concat([df1, new_X], axis=0).reset_index(drop=True)
    LSDMLO_target = pd.concat([df2, target], axis=0).reset_index(drop=True)
    return LSDMLO_new_X,LSDMLO_target
