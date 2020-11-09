import numpy as np
import random

def PLA(x,y,w):
    count = 0
    update_count = 0
    while count!= 500:
        random_num = random.randrange(0,100,1) #random_number(0~99)

        h = np.sign(np.dot(w.T,x[random_num])) #計算h

        if h == 0: #若sign(h)=0 則把h變成-1
            h = -1

        if h != y[random_num]: #若跟labeled的不一樣，則更新w
            w += y[random_num]*x[random_num]
            count = 0
            update_count += 1
        else:
            count+=1
        
    return w,update_count

def Problem_16_17(data):
    update_list = []
    w0 = []
    #Experiment
    for j in range(1000):
        #load_data
        x = np.empty((100,11)) # features
        y = np.empty(100)   # label
        w = np.zeros(11)    # Wt
        wpla = np.empty(11) # Wf
        for idx,i in enumerate(data):
            x[idx][0] = 1
            x[idx][1:] = i[0:-1]
            y[idx] = i[-1]
        # PLA
        wpla,update_count = PLA(x,y,w) # (11,1)
        update_list.append(update_count)
        w0.append(wpla[0])

    update_list = np.array(update_list)
    print('Problem_16: ',int(np.median(update_list)))
    w0 = np.array(w0)
    print('Problem_17:',int(np.median(w0)))

def Problem_18(data):
    update_list = []
    #Experiment
    for j in range(1000):
        #load_data
        x = np.empty((100,11)) # features
        y = np.empty(100)   # label
        w = np.zeros(11)    # Wt
        wpla = np.empty(11) # Wf
        for idx,i in enumerate(data):
            x[idx][0] = 10
            x[idx][1:] = i[0:-1]
            y[idx] = i[-1]
        # PLA
        wpla,update_count = PLA(x,y,w) # (11,1)
        update_list.append(update_count)

    update_list = np.array(update_list)
    print('Problem_18: ',int(np.median(update_list)))

def Problem_19(data):
    update_list = []
    #Experiment
    for j in range(1000):
        #load_data
        x = np.empty((100,11)) # features
        y = np.empty(100)   # label
        w = np.zeros(11)    # Wt
        wpla = np.empty(11) # Wf
        for idx,i in enumerate(data):
            x[idx][0] = 0
            x[idx][1:] = i[0:-1]
            y[idx] = i[-1]
        # PLA
        wpla,update_count = PLA(x,y,w) # (11,1)
        update_list.append(update_count)

    update_list = np.array(update_list)
    print('Problem_19: ',int(np.median(update_list)))

def Problem_20(data):
    update_list = []
    #Experiment
    for j in range(1000):
        #load_data
        x = np.empty((100,11)) # features
        y = np.empty(100)   # label
        w = np.zeros(11)    # Wt
        wpla = np.empty(11) # Wf
        for idx,i in enumerate(data):
            x[idx][0] = 0
            x[idx][1:] = 1/4 * i[0:-1]
            y[idx] = i[-1]
        # PLA
        wpla,update_count = PLA(x,y,w) # (11,1)
        update_list.append(update_count)

    update_list = np.array(update_list)
    print('Problem_20: ',int(np.median(update_list)))
if __name__ == "__main__":
    
    data = np.loadtxt('./hw1_train.txt')
    Problem_16_17(data)
    Problem_18(data)
    Problem_19(data)
    Problem_20(data)
