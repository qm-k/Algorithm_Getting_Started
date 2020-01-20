import cv2
import time
import numpy as np
from random import randint
from rknn.api import RKNN
from multiprocessing import Process, Queue, Lock
import multiprocessing

nPoints = 18
# COCO Output Format
keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
              [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
              [1,0], [0,14], [14,16], [0,15], [15,17],
              [2,16], [5,17] ]

# index of pafs correspoding to the POSE_PAIRS
# e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44],
          [19,20], [21,22], [23,24], [25,26], [27,28], [29,30],
          [47,48], [49,50], [53,54], [51,52], [55,56],
          [37,38], [45,46]]

colors = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
         [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
         [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]


def getKeypoints(probMap, threshold=0.1):

    mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)

    mapMask = np.uint8(mapSmooth>threshold)
    #np.set_printoptions(threshold=np.inf)
    keypoints = []

    #find the blobs
    _, contours, hierarchy = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #for each blob find the maxima
    #对于每个关键点，对confidence map 应用一个阀值（本例采用0.1），生成二值图。
    #首先找出每个关键点区域的全部轮廓。
    #生成这个区域的mask。
    #通过用probMap乘以这个mask，提取该区域的probMap。
    #找到这个区域的本地极大值。要对每个即关键点区域进行处理。
    #本地极大值对应的坐标就是关键点坐标
    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints


# Find valid connections between the different joints of a all persons present
def getValidPairs(output,detected_keypoints):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7
    #detected_keypoints = []##尝试
    # loop for every POSE_PAIR
    for k in range(len(mapIdx)):
        # A->B constitute a limb
        pafA = output[0, mapIdx[k][0], :, :]
        pafB = output[0, mapIdx[k][1], :, :]
        pafA = cv2.resize(pafA, (frameWidth, frameHeight))
        pafB = cv2.resize(pafB, (frameWidth, frameHeight))

        # candA: (124, 365, 0.17102814, 43)
        #                               detected_keypoints keypoint_id
        # Find the keypoints for the first and second limb
        #把连接对上的关键点提取出来，相同的关键点放一起。把关键点对分开地方到两个列表上
        #（列表名为candA和candB）。在列表candA上的每一个点都会和列表candB上某些点连接
        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]

        nA = len(candA)
        nB = len(candB)

        # If keypoints for the joint-pair is detected
        # check every joint in candA with every joint in candB
        # Calculate the distance vector between the two joints
        # Find the PAF values at a set of interpolated points between the joints
        # Use the above formula to compute a score to mark the connection valid

        if( nA != 0 and nB != 0):
            valid_pair = np.zeros((0,3))
            for i in range(nA):
                max_j=-1
                maxScore = -1
                found = 0
                for j in range(nB):
                    # Find d_ij
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    # Find p(u)
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                    # Find L(p(u))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ])
                    # Find E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores)/len(paf_scores)

                    # Check if the connection is valid
                    # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
                    if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1
                # Append the connection to the list
                if found:
                    #   detected_keypoints keypoint_id
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)
            # Append the detected connections to the global list
            valid_pairs.append(valid_pair)
        else: # If no keypoints are detected
            invalid_pairs.append(k)
            valid_pairs.append([])
    return valid_pairs, invalid_pairs



# This function creates a list of keypoints belonging to each person
# For each detected valid pair, it assigns the joint(s) to a person
def getPersonwiseKeypoints(valid_pairs, invalid_pairs,keypoints_list):
    # the last number in each row is the overall score

    #我们首先创建空列表，用来存放每个人的关键点（即关键部位）
    personwiseKeypoints = -1 * np.ones((0, 19))
    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:,0]
            partBs = valid_pairs[k][:,1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                #遍历每一个连接对，检查连接对中的partA是否已经存在于任意列表之中
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                #如果存在，那么意味着这关键点属于当前列表，同时连接对中的partB也同样属于这个人体
                #把连接对中的partB增加到partA所在的列表。
                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                # if find no partA in the subset, create a new subset
                #如果partA不存在于任意列表，那么说明这一对属于一个还没建立列表的人体，于是需要新建一个新列表。
                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    # add the keypoint_scores for the two keypoints and the paf_score
                    row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    return personwiseKeypoints


inWidth = 368
inHeight = 368

frameWidth = 416
frameHeight = 416




def load_model():
        rknn = RKNN()
        print('-->loading model')
        rknn.load_rknn('./pose_deploy_linevec_pre_compile.rknn')
        print('loading model done')

        print('--> Init runtime environment')
        ret = rknn.init_runtime()
        if ret != 0:
                print('Init runtime environment failed')
                exit(ret)
        print('!!!!!!!!!!!!!!!!!_____________________done___________________!!!!!!!!!!!!!!!!!!!!')
        return rknn




def video_capture(src, q_frame:Queue, q_image:Queue):
    #VideoCapture
    video = cv2.VideoCapture(int(src))

    
    while True:
        s = time.time()
        hasFrame, frame = video.read()



        assert hasFrame, 'read video frame failed.'
        print('capture read used {} ms.'.format((time.time() - s) * 1000))


        s = time.time()
        image = cv2.resize(frame, (368, 368))
        ##image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#处理待定
        print('capture resize used {} ms.'.format((time.time() - s) * 1000))
        ##frameWidth = frame.shape[1]
        ##frameHeight = frame.shape[0]



        s = time.time()
        if q_frame.empty():
            q_frame.put(frame)
        
        if q_image.full():
            print("q_img is full")
            continue
        else:
            q_image.put(image)
        
        print("capture put to queue used {} ms".format((time.time()-s)*1000))
###!~!!!!!!!!之后的程序换成image进行处理




def infer_rknn(q_image:Queue, q_infer:Queue):
    rknn = load_model()
    ###rknn.get_sdk_version()

    while True:
        s = time.time()
        print("ready to get img")
        image = q_image.get()
        print('Infer get, used time {} ms. '.format((time.time() - s) * 1000))

        s = time.time()


        ###!!!!!!!!!!!!!!!!!!!!!!!###
        ########rknn.inference#######
        ###!!!!!!!!!!!!!!!!!!!!!!!###



        frame_input = np.transpose(image, [2, 0, 1])
        print("----------------frame_input type is :",type(frame_input))
        print("================the value of frame_input is :",frame_input)
        t = time.time()
        #[output] = rknn.inference(inputs=[frame_input], data_format="nchw")
        [output] = rknn.inference(inputs=[frame_input], data_format="nchw")
        print("time:", time.time()-t)
        print("----------------output type is :",type(output))
        print("================the value of output is :",output)
        #print("============transfor nonetype to array begin===========")
        #output = np.array(output)
        #print("-----------------transfor is OK--------------------")
        output = output.reshape(1, 57, 46, 46)

        q_infer.put(output)
        print("==========infer done ,have cast :",time.time()-t)


def post_process(q_infer,q_image, q_objs):#, q_objs
    while True:
        print("begin to post_process=========================")
        output = q_infer.get()
        frame = q_image.get()
        detected_keypoints = [] #///////////原始位置
        keypoints_list = np.zeros((0,3))
        keypoint_id = 0
        threshold = 0.1


        for part in range(nPoints):
            probMap = output[0,part,:,:]
            probMap = cv2.resize(probMap, (frame.shape[1], frame.shape[0]))
            keypoints = getKeypoints(probMap, threshold)
            keypoints_with_id = []
            for i in range(len(keypoints)):

                keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                keypoint_id += 1

            detected_keypoints.append(keypoints_with_id)


        frameClone = frame.copy()
        
        valid_pairs, invalid_pairs = getValidPairs(output,detected_keypoints)
        personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs,keypoints_list)


        #连接各个人体关键点
        for i in range(17):
            for n in range(len(personwiseKeypoints)):
                index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                if -1 in index:
                    continue
                B = np.int32(keypoints_list[index.astype(int), 0])
                A = np.int32(keypoints_list[index.astype(int), 1])
                cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)
        
        q_objs.put(frameClone) 


q_frame = Queue(maxsize=1)
q_image = Queue(maxsize=3)
q_infer = Queue(maxsize=3)
q_objs = Queue(maxsize=3)


p_cap1 = Process(target=video_capture, args=('0', q_frame, q_image))
p_cap2 = Process(target=video_capture, args=('0', q_frame, q_image))
p_infer1 = Process(target=infer_rknn, args=(q_image, q_infer))
p_infer2 = Process(target=infer_rknn, args=(q_image, q_infer))
p_post1 = Process(target=post_process, args=(q_infer,q_image, q_objs))#, q_objs
p_post2 = Process(target=post_process, args=(q_infer,q_image, q_objs))#, q_objs

p_cap1.start()
p_cap2.start()
p_infer1.start()
p_infer2.start()
p_post1.start()
p_post2.start()


for p in multiprocessing.active_children():
        print("child   p.name:" + p.name + "\tp.id" + str(p.pid))


while True:
    frameClone = q_objs.get()
    print('read to show')
    cv2.imshow("Detected Pose" , frameClone)
    print(frameClone)
    #sp = frameClone.shape()
    #print("=========================img_shape====================================")
    #print(sp)
    #print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    c = cv2.waitKey(5) & 0xff
    if c == 27:
        cv2.destroyAllWindows() 
        break

#rknn.release()
p_cap1.terminate()
p_cap2.terminate()
p_infer1.terminate()
p_infer2.terminate()
p_post1.terminate()
p_post2.terminate()
exit()