import os, torch, numpy, cv2, random, glob, python_speech_features
import pandas as pd
from scipy.io import wavfile
from torchvision.transforms import RandomCrop


def load_face_sequence(data, dataPath, numFrames, sizeVideoInput, faceAug): 
    dataName = data[0]
    videoName = data[0][: 12 + data[0][12 :].index('_')]
    faceFolderPath = os.path.join(dataPath, videoName, dataName)
    faceFiles = glob.glob("%s/*.jpg"%faceFolderPath)
    sortedFaceFiles = sorted(faceFiles, key=lambda data: (float(data.split('/')[-1][:-4])), reverse=False) 
    faces = []
    H = sizeVideoInput
    if faceAug == True:
        new = int(H*random.uniform(0.7, 1))
        x, y = numpy.random.randint(0, H - new), numpy.random.randint(0, H - new)
        M = cv2.getRotationMatrix2D((H/2,H/2), random.uniform(-15, 15), 1)
        augType = random.choice(['orig', 'flip', 'crop', 'rotate']) 
    else:
        augType = 'orig'
    for faceFile in sortedFaceFiles[:numFrames]:
        face = cv2.imread(faceFile)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (H,H))
        if augType == 'orig':
            faces.append(face)
        elif augType == 'flip':
            faces.append(cv2.flip(face, 1))
        elif augType == 'crop':
            faces.append(cv2.resize(face[y:y+new, x:x+new] , (H,H))) 
        elif augType == 'rotate':
            faces.append(cv2.warpAffine(face, M, (H,H)))
    faces = numpy.array(faces)
    return faces


def generate_audio_set(dataPath, batchList):
    audioSet = {}
    for line in batchList:
        data = line.split('\t')
        videoName = data[0][: 12 + data[0][12 :].index('_')]
        dataName = data[0]
        _, audio = wavfile.read(os.path.join(dataPath, videoName, dataName + '.wav'))
        audioSet[dataName] = audio
    return audioSet


def overlap(dataName, audio, audioSet):
    sample = random.sample(list(set(list(audioSet.keys())) - {dataName}), 1)
    noiseName =  sample[0] #random.sample(list(set(list(audioSet.keys())) - {dataName}), 1)[0]
    noiseAudio = audioSet[noiseName]    
    snr = [random.uniform(-5, 5)]
    if len(noiseAudio) < len(audio):
        shortage = len(audio) - len(noiseAudio)
        noiseAudio = numpy.pad(noiseAudio, (0, shortage), 'wrap')
    else:
        noiseAudio = noiseAudio[:len(audio)]
    noiseDB = 10 * numpy.log10(numpy.mean(abs(noiseAudio ** 2)) + 1e-4)
    cleanDB = 10 * numpy.log10(numpy.mean(abs(audio ** 2)) + 1e-4)
    noiseAudio = numpy.sqrt(10 ** ((cleanDB - noiseDB - snr) / 10)) * noiseAudio
    audio = audio + noiseAudio    
    return audio.astype(numpy.int16)


def load_audio(data, dataPath, numFrames, audioAug, audioSet = None):
    dataName = data[0]
    fps = float(data[2])    
    audio = audioSet[dataName]    
    if audioAug == True and len(audioSet) > 1:
        augType = random.randint(0,1)
        if augType == 1:
            audio = overlap(dataName, audio, audioSet)
        else:
            audio = audio
    # fps is not always 25, in order to align the face, we modify the window and step in MFCC extraction process based on fps
    # print(f"Processing audio from video with {fps} frames per second, with length {len(audio)} at 16 kHz, window length of {0.025 * 25 / fps} and window step of {0.010 * 25 / fps}")
    audio = python_speech_features.mfcc(audio, 16000, nfft = 1024, numcep = 13, winlen = 0.025 * 25 / fps, winstep = 0.010 * 25 / fps)
    maxAudio = int(numFrames * 4)
    if audio.shape[0] < maxAudio:
        shortage    = maxAudio - audio.shape[0]
        audio     = numpy.pad(audio, ((0, shortage), (0,0)), 'wrap')
    audio = audio[:int(round(numFrames * 4)),:]  
    return audio


def load_body_joint_keypoints(data, df_kp, numFrames, upperBody, bodyPoseAug, bodyPoseAugProb = 0.1, bodyPoseAugMag = 0.01):
    dataName = data[0]
    keypoint_list = range(11 if upperBody else 17)
    axes = ["x", "y", "score"]
    columns = ["kp" + ("0" + str(idx))[-2 :] + "_" + axis + ("_norm" if axis != "score"  else "") for idx in keypoint_list for axis in axes]
    body_joint_keypoint_data =  df_kp[df_kp["entity_id"] == dataName][columns].to_numpy()[: numFrames, :].reshape(-1, len(keypoint_list), len(axes))
    
    if bodyPoseAug:
        rnd_thr = numpy.random.rand(*body_joint_keypoint_data.shape[: 2])
        r = numpy.sqrt(numpy.random.rand(*body_joint_keypoint_data.shape[: 2])) * bodyPoseAugMag
        theta = numpy.random.rand(*body_joint_keypoint_data.shape[: 2]) * 2 * numpy.pi
        x = r * numpy.cos(theta)
        y = r * numpy.sin(theta)
        x = numpy.where(rnd_thr < bodyPoseAugProb, x, 0.)
        y = numpy.where(rnd_thr < bodyPoseAugProb, y, 0.)
        if len(axes) == 2:
            aug = numpy.stack([x, y], axis = 2)
        elif len(axes) == 3:
            aug = numpy.stack([x, y, numpy.zeros(body_joint_keypoint_data.shape[: 2])], axis = 2)
        else:
            raise Exception("Unexpected behaviour.")
        body_joint_keypoint_data += aug
    body_joint_keypoint_data = numpy.array(body_joint_keypoint_data)
    
    return body_joint_keypoint_data


def load_label(data, numFrames):
    res = []
    labels = data[3].replace('[', '').replace(']', '')
    labels = labels.split(',')
    for label in labels:
        res.append(int(label))
    res = numpy.array(res[:numFrames])
    return res


class train_loader(object):
    def __init__(self, trialFileName, keypointsFileName, facePath, audioPath, sizeVideoInput, batchSize, bodyPose = True, upperBody = True, bodyPoseAugProb = 0.1, bodyPoseAugMag = 0.01, **kwargs):
        self.audioPath  = audioPath
        self.facePath = facePath
        self.sizeVideoInput = sizeVideoInput
        
        self.bodyPose = bodyPose
        if self.bodyPose:
            self.bodyPoseAugProb = bodyPoseAugProb
            self.bodyPoseAugMag = bodyPoseAugMag
            self.upperBody = upperBody
            self.keypoint_df = pd.read_csv(keypointsFileName, sep = ",")
        
        self.miniBatch = []      
        mixLst = open(trialFileName).read().splitlines()
        
        # sort the training set by the length of the videos, shuffle them to make more videos in the same batch belong to different movies
        sortedMixLst = sorted(mixLst, key=lambda data: (int(data.split('\t')[1]), int(data.split('\t')[-1])), reverse=True)
        
        start = 0
        while True:
            length = int(sortedMixLst[start].split('\t')[1])
            end = min(len(sortedMixLst), start + max(int(batchSize / length), 1))
            self.miniBatch.append(sortedMixLst[start:end])
            if end == len(sortedMixLst):
                break
            start = end

    def __getitem__(self, index):
        batchList    = self.miniBatch[index]
        numFrames   = int(batchList[-1].split('\t')[1])
        faceFeatures, audioFeatures, bodyFeatures, labels = [], [], [], []
        audioSet = generate_audio_set(self.audioPath, batchList) # load the audios in this batch to do augmentation
        for line in batchList:
            data = line.split('\t')
            faceFeatures.append(load_face_sequence(data, self.facePath, numFrames, self.sizeVideoInput, faceAug = True))
            audioFeatures.append(load_audio(data, self.audioPath, numFrames, audioAug = True, audioSet = audioSet))
            labels.append(load_label(data, numFrames))
            
            if self.bodyPose:
                bodyFeatures.append(load_body_joint_keypoints(data, self.keypoint_df, numFrames, upperBody = self.upperBody, bodyPoseAug = True, bodyPoseAugProb = self.bodyPoseAugProb, bodyPoseAugMag = self.bodyPoseAugMag))
        
        audioFeatures = torch.FloatTensor(numpy.array(audioFeatures))
        faceFeatures = torch.FloatTensor(numpy.array(faceFeatures))
        labels = torch.LongTensor(numpy.array(labels))
        
        if self.bodyPose:
            bodyFeatures = torch.FloatTensor(numpy.array(bodyFeatures))
        return faceFeatures, audioFeatures, bodyFeatures, labels

    def __len__(self):
        return len(self.miniBatch)


class val_loader(object):
    def __init__(self, trialFileName, keypointsFileName, facePath, audioPath, sizeVideoInput, bodyPose = True, upperBody = True, **kwargs):
        self.audioPath  = audioPath
        self.facePath = facePath
        self.sizeVideoInput = sizeVideoInput
        self.miniBatch = open(trialFileName).read().splitlines()
        
        self.bodyPose = bodyPose
        if self.bodyPose:
            self.upperBody = upperBody
            self.keypoint_df = pd.read_csv(keypointsFileName, sep = ",")

    def __getitem__(self, index):
        line       = [self.miniBatch[index]]
        numFrames  = int(line[0].split('\t')[1])
        audioSet   = generate_audio_set(self.audioPath, line)
        data = line[0].split('\t')
        faceFeatures = [load_face_sequence(data, self.facePath,numFrames, self.sizeVideoInput, faceAug = False)]
        audioFeatures = [load_audio(data, self.audioPath, numFrames, audioAug = False, audioSet = audioSet)]
        labels = [load_label(data, numFrames)]
        
        audioFeatures = torch.FloatTensor(numpy.array(audioFeatures))
        faceFeatures = torch.FloatTensor(numpy.array(faceFeatures))
        labels = torch.LongTensor(numpy.array(labels))
        
        if self.bodyPose:
            bodyFeatures = [load_body_joint_keypoints(data, self.keypoint_df, numFrames, upperBody = self.upperBody, bodyPoseAug = False)]
            bodyFeatures = torch.FloatTensor(numpy.array(bodyFeatures))
        else:
            bodyFeatures = []
        return faceFeatures, audioFeatures, bodyFeatures, labels

    def __len__(self):
        return len(self.miniBatch)
