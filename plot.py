import pickle
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from numpy.random import rand

'''
def goalGanReadv1(file_path,scale = 1, r_min = 0.1):
    with open(file_path, "rb") as f:
        pickle_in = pickle.load(f)
    hotspot = np.zeros((height * int(1 / scale) , width * int(1 / scale) ))
    freq = np.zeros_like(hotspot)
    coverageHistory = {"iteration": [], "coverage": []}
    rewards = []
    X = []
    Y = []
    for each in pickle_in.values():
        rewards += list(each[2])

    for iteration, result in pickle_in.items():

        for i, step in enumerate(result):
            x, y = step[0], step[1]
            if abs(x) > 3:
                continue
            if abs(x) > 3:
                continue
            x_scale, y_scale = int(round((x + pad) / scale)), int(round((y + pad) / scale))
            freq[x_scale][y_scale] += 1
            if (step[2]) == 1:
                hotspot[x_scale][y_scale] += 1
        coverageHistory["iteration"].append(iteration)
        coverageHistory["coverage"].append(sum(sum(divide(hotspot, freq) > r_min)) / hotspot.size)
        if iteration % 100 == 0:
            HotSpotPlot(divide(hotspot,freq))
    return  divide(hotspot, freq), coverageHistory



def goalGanReadv2(file_path,scale = 1, r_min = 0.1):
    with open(file_path, "rb") as f:
        pickle_in = pickle.load(f)
    hotspot = np.zeros((height * int(1 / scale) , width * int(1 / scale) ))
    freq = np.zeros_like(hotspot)
    coverageHistory = {"iteration": [], "coverage": []}
    rewards = []
    X = []
    Y = []
    for each in pickle_in.values():
        rewards += list(each[2])

    for iteration, result in pickle_in.items():
        for i, goal in enumerate(result[2]):
            # if goal < r_min:
            # continue
            x, y = result[0][i]
            X.append(x)
            Y.append(y)
    for iteration, result in pickle_in.items():

        for i, goal in enumerate(result[2]):
            # if goal < r_min:
            # continue
            x, y = result[0][i]
            if abs(x) > 5:
                continue
            if abs(x) > 5:
                continue
            x_scale, y_scale = int(round((x + pad) / scale)), int(round((y + pad) / scale))
            hotspot[x_scale][y_scale] = max(hotspot[x_scale][y_scale], goal)
            #freq[x_scale][y_scale] += 1
        coverageHistory["iteration"].append(iteration)
        coverageHistory["coverage"].append(sum(sum(hotspot > r_min)) / hotspot.size)
        # if iteration % 100 == 0:
        # HotSpotPlot(divide(hotspot,freq))
    return  hotspot, coverageHistory


def divide(a,b):
    row, column = a.shape
    res = np.zeros_like(a)
    for i in range(row):
        for j in range(column):
            if b[i][j] == 0:
                continue
            else:
                res[i][j] = a[i][j] / b[i][j]
    return res

'''

def readBaseLineInput(file_path, r_min, scale = 1):
    C_total = []
    H = []
    for path in file_path:

        with open(path, "rb") as f:
            pickle_in = pickle.load(f)
        hotspot = np.zeros((height * int(1 / scale), width * int(1 / scale)))
        freq = np.zeros_like(hotspot)
        coverageHistory = {"iteration": [], "coverage": []}

        rewards = []
        for each in pickle_in.values():
            rewards += list(each[2])
        for iteration, result in pickle_in.items():

            for i, goal in enumerate(result[2]):
                #if goal < r_min:
                    #continue
                x, y = result[0][i]
                if abs(x) > pad:
                    continue
                if abs(x) > pad:
                    continue
                x_scale, y_scale = int(((x + pad) / scale)), int(((y + pad) / scale))
                hotspot[x_scale][y_scale] = max(hotspot[x_scale][y_scale],goal)
                #freq[x_scale][y_scale] += 1
            coverageHistory["iteration"].append(iteration)
            coverageHistory["coverage"].append(sum(sum(hotspot > r_min)) / hotspot.size)
            #if iteration % 100 == 0:
                #HotSpotPlot(divide(hotspot,freq))
        #H.append(divide(hotspot,freq))
        H.append(hotspot)
        C_total.append(coverageHistory)


    H_ave = np.zeros((height * int(1 / scale), width * int(1 / scale)))
    C_iteration = []
    C_coverage = []
    for i, h in enumerate(H):
        H_ave += h
        C_iteration += C_total[i]["iteration"]
        C_coverage += C_total[i]["coverage"]
    H_ave = H_ave / len(H)

    C_dict = {"iteration": C_iteration, "coverage":C_coverage}

    return H_ave, C_dict


'''

def readBaseLineL2(file_path,r_min):
    C_total = []
    H = []
    for path in file_path:
        with open(path, "rb") as f:
            pickle_in = pickle.load(f)
        hotspot = np.zeros([height, width])
        freq = np.zeros_like(hotspot)
        coverageHistory = {"iteration":[], "coverage":[]}
        rewards = []
        length = []
        for iteration, result in pickle_in.items():
            rewards += list(result[2])
            length.append(len(result[2]))
        #normalized_reward = preprocessing.scale(np.array(rewards))
        min_reward = min(rewards)
        max_reward = max(rewards)
        diff = max_reward - min_reward
        normalized_reward = list(map(lambda x: (x- min_reward) / diff, rewards))

        index = -1
        for iteration, result in pickle_in.items():
            goals = result[0]
            for i in range(len(goals)):
                index += 1
                x,y = goals[i]
                x, y = int(x+pad), int(y+pad)
                #if normalized_reward[index] >= r_min:
                hotspot[x,y] +=  normalized_reward[index]
                freq[x,y] += 1
            coverageHistory["iteration"].append(iteration)
            coverageHistory["coverage"].append(   sum(sum(divide(hotspot, freq) > r_min)) / hotspot.size)

                #HotSpotPlot(divide(hotspot,freq))
        H.append(divide(hotspot,freq))
        C_total.append(coverageHistory)

    H_ave = np.zeros((height * int(1 / 1), width * int(1 / 1)))
    C_iteration = []
    C_coverage = []
    for i, h in enumerate(H):
        H_ave += h
        C_iteration += C_total[i]["iteration"]
        C_coverage += C_total[i]["coverage"]
    H_ave = H_ave / len(H)

    C_dict = {"iteration": C_iteration, "coverage": C_coverage}

    return H_ave, C_dict
'''

def HotSpotPlot(hotspot):
    #print(hotspot)
    plt.imshow(hotspot)
    plt.grid(True)
    plt.colorbar()
    plt.show()

def LRplot(history_list, label_list):
    for i, history in enumerate(history_list):
        df = pd.DataFrame(history)
        ax = sns.regplot(x="iteration", y="coverage", data=df, dropna=True, label=label_list[i])
    ax.legend(loc="best")
    plt.show()


def goalTransfer(file_path, r_min = 0.3, r_max = 0.6):
    with open(file_path, "rb") as f:
        pickle_in = pickle.load(f)


    goal_total = [[],[],[]]
    for iteration, result in pickle_in.items():

        for i, goal in enumerate(result[2]):
            # if goal < r_min:
            # continue
            x, y = result[0][i]
            if abs(x) > pad:
                continue
            if abs(x) > pad:
                continue
            if i > 8:
                continue
            x_scale, y_scale = x + pad, y + pad
            if goal < r_min:
                goal_total[0].append((x_scale,y_scale))
            elif  goal >= r_min and goal <= r_max:
                goal_total[1].append((x_scale, y_scale))
            elif goal > r_max:
                goal_total[2].append((x_scale, y_scale))
        if (iteration +1) % 100 == 0:
            fig, ax = plt.subplots()
            color = ['red','blue','green']
            for i, label in enumerate(['low rewards', 'good goals', 'high rewards']):
                x, y = [height-each[0]   for each in goal_total[i]], [each[1] for each in goal_total[i]]
                ax.scatter(y, x, c=color[i], label=label,
                           alpha=0.3, edgecolors='none')

            ax.legend()
            ax.grid(True)

            plt.xlim(-0.5, 10.5)
            plt.ylim(-0.5, 10.5)
            plt.show()
            goal_total = [[],[],[]]




if __name__ == "__main__":

    global width
    global height
    width = 11
    height = 11
    global pad
    pad = 5
    #_, hotspot, coverageHistory = readInput(pickle_in, r_min=0.92)
    baseline_1 = "dict.0.5_records_BL_One_0"
    baseline_2 = "dict.records_BL_Two_Modified_3"
    goal_gan = "dict.Goal_GAN_fixed_1-5-2"


    from os import listdir
    from os.path import isfile, join
    mypath = "/home/danteluo/Documents/2017Fall/EECS598/Project/Goal_GAN"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    baseline1_file = [file for file in onlyfiles if baseline_1 in file]
    baseline2_file = [file for file in onlyfiles if baseline_2 in file]
    goal_gan_file = [file for file in onlyfiles if goal_gan in file]

    # goal_gan251_file = [file for file in onlyfiles if goal_gan251 in file]
    # goal_gan252_file = [file for file in onlyfiles if goal_gan252 in file]
    # goal_gan152_file = [file for file in onlyfiles if goal_gan152 in file]
    # goal_gan281_file = [file for file in onlyfiles if goal_gan281 in file]
    # goal_gan365_file = [file for file in onlyfiles if goal_gan365 in file]
    #
    #
    # _, coverageHistory_251 = readBaseLineInput(goal_gan251_file,r_min=0.3)
    # _, coverageHistory_252 = readBaseLineInput(goal_gan252_file,r_min=0.3)
    # _, coverageHistory_152 = readBaseLineInput(goal_gan152_file,r_min=0.3)
    # _, coverageHistory_281 = readBaseLineInput(goal_gan281_file,r_min=0.3)
    # _, coverageHistory_365 = readBaseLineInput(goal_gan365_file,r_min=0.3)

    # label_list = ["251", "252", "152","281","365"]

    goalTransfer(goal_gan)

    hotspot, coverageHistory = readBaseLineInput(goal_gan_file, r_min=0.2)

    _, coverageHistory_b1 = readBaseLineInput(baseline1_file,r_min=0.2)
    _, coverageHistory_b2 = readBaseLineInput(baseline2_file,r_min=0.2)
    HotSpotPlot(hotspot)
    label_list = ["Goal_GAN","UniformSampling","UniformSampling_L2Loss"]
    LRplot([coverageHistory,coverageHistory_b1,coverageHistory_b2], label_list)


