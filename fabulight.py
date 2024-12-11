import time, argparse, warnings
import glob, pickle
import random, os, shutil
import numpy as np
import torch
import torch.nn.functional as F

from dataLoader import train_loader, val_loader
from ASD import ASD

# For the demo
import cv2
from rtmlib import PoseTracker, Body, draw_skeleton
import sounddevice as sd
import python_speech_features
import tqdm


def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def execute_demo(asd_model, sizeVideoInput, bodyPose, upperBody, pose_recognising_min_score = 0.8, **kwargs):
    device = 'cuda'
    backend = 'onnxruntime'  # opencv, onnxruntime, openvino
    openpose_skeleton = False  # True for openpose-style, False for mmpose-style

    tracker = PoseTracker(
        Body,
        det_frequency=1,
        tracking_thr=0.3,
        to_openpose=openpose_skeleton,
        mode='performance',  # balanced, performance, lightweight
        backend=backend,
        device=device)
    
    audio_sr = 16000
    chunk_size = 0.1  # Audio chunk size in seconds (100ms)
    audio_stream_blocksize = int(chunk_size * audio_sr)
    max_win_size = 120
    speech_activity_acceptance_thr = 0.5

    asd_model.eval()
    asd_model.to(device)
    audio_buffer = np.array([])

    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the maximum audio buffer size to align with the input window size of the model.
    # The buffer accommodates four times the video frames, adjusted for frame rate (fps) and chunk size (100 ms)
    max_audio_buffer_size = int(np.ceil(4 * max_win_size * 0.1  * 25 / fps * audio_stream_blocksize))

    def audio_callback(indata, frames, time, status):
        nonlocal audio_buffer, max_win_size, max_audio_buffer_size
        
        new_data = indata[:, 0].flatten()
        audio_buffer = np.concatenate((audio_buffer, new_data))
        
        if audio_buffer.shape[0] > max_audio_buffer_size:
            audio_buffer = audio_buffer[-max_audio_buffer_size :]

    audio_stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=audio_sr, blocksize=audio_stream_blocksize)
    audio_stream.start()
            
    spk_data = dict()
    frame_idx = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break
        frame_idx += 1
        start_time = time.time()
        
        frame_width, frame_height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        keypoints, scores = tracker(frame)
        speakers_in_last_frame = []
        map_num_frames_to_spk_id = dict()
        for spk_id, spk_body_bbox, kp_data, kp_score in zip(tracker.track_ids_last_frame, tracker.bboxes_last_frame, keypoints, scores):
            if max(kp_score) >= pose_recognising_min_score:
                # Deduce head bounding box from head keypoints
                face_xc = sum(kp_data[: 5, 0]) / 5
                face_yc = sum(kp_data[: 5, 1]) / 5
                face_x0 = min(kp_data[: 5, 0])
                face_y0 = min(kp_data[: 5, 1])
                face_x1 = max(kp_data[: 5, 0])
                face_y1 = max(kp_data[: 5, 1])
                
                max_d = max(face_xc - face_x0, face_yc - face_y0, face_x1 - face_xc, face_y1 - face_yc)
                face_x0 = int(np.round(face_xc - max_d))
                face_y0 = int(np.round(face_yc - max_d))
                face_x1 = int(np.round(face_xc + max_d))
                face_y1 = int(np.round(face_yc + max_d))
                
                face_x0 = max(0, face_x0)
                face_y0 = max(0, face_y0)
                face_x1 = min(frame_width, face_x1)
                face_y1 = min(frame_height, face_y1)
                
                if face_x1 - face_x0 <= 0 or face_y1 - face_y0 <= 0:
                    continue
                
                if spk_id not in spk_data:
                    spk_data[spk_id] = {"kp_noncentral" : [], "kp_central" : [], "face_bbox" : [], "face_crop" : [], "audio": None, "speaking" : None}
                
                spk_data[spk_id]["face_bbox"].append(np.array([face_x0, face_y0, face_x1, face_y1], dtype = np.int16))
                
                face_crop = frame[spk_data[spk_id]["face_bbox"][-1][1] : spk_data[spk_id]["face_bbox"][-1][3], spk_data[spk_id]["face_bbox"][-1][0] : spk_data[spk_id]["face_bbox"][-1][2]]
                face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                face_crop = cv2.resize(face_crop, (sizeVideoInput, sizeVideoInput))
                spk_data[spk_id]["face_crop"].append(face_crop)
                
                spk_body_bbox_center = ((spk_body_bbox[0] + spk_body_bbox[2]) / 2, (spk_body_bbox[1] + spk_body_bbox[3]) / 2)
                spk_body_bbox_width = spk_body_bbox[2] - spk_body_bbox[0]
                spk_body_bbox_height = spk_body_bbox[3] - spk_body_bbox[1]
                
                spk_data[spk_id]["kp_noncentral"].append(np.array([[kpx, kpy, score] for (kpx, kpy), score in zip(kp_data, kp_score)], dtype=np.float32))
                spk_data[spk_id]["kp_central"].append(np.array([[(kpx - spk_body_bbox_center[0]) / spk_body_bbox_width, (kpy - spk_body_bbox_center[1]) / spk_body_bbox_width, score] for (kpx, kpy), score in zip(kp_data, kp_score)], dtype=np.float32))
                
                if len(spk_data[spk_id]["face_bbox"]) > max_win_size:
                    spk_data[spk_id]["face_bbox"] = spk_data[spk_id]["face_bbox"][-max_win_size :]
                    spk_data[spk_id]["kp_noncentral"] = spk_data[spk_id]["kp_noncentral"][-max_win_size :]
                    spk_data[spk_id]["kp_central"] = spk_data[spk_id]["kp_central"][-max_win_size :]
                    spk_data[spk_id]["face_crop"] = spk_data[spk_id]["face_crop"][-max_win_size :]
                
                num_frames = len(spk_data[spk_id]["face_bbox"])
                map_num_frames_to_spk_id[num_frames] = map_num_frames_to_spk_id.get(num_frames, []) + [spk_id]
                speakers_in_last_frame.append(spk_id)
        
        det_time = time.time() - start_time
        #print('det after extracting video features: ', det_time)
        
        full_audio_mfcc = python_speech_features.mfcc(audio_buffer, 16000, nfft = 1024, numcep = 13, winlen = 0.25 * chunk_size * 25 / fps, winstep = 0.1 * chunk_size * 25 / fps)
        
        for num_frames in map_num_frames_to_spk_id:
            spk_audio = full_audio_mfcc[:]
            maxAudio = int(num_frames * 4)
            if full_audio_mfcc.shape[0] < maxAudio:
                shortage  = maxAudio - spk_audio.shape[0]
                spk_audio = np.pad(spk_audio, ((0, shortage), (0,0)), 'wrap')
            spk_audio = spk_audio[: int(round(num_frames * 4)), :]
            for spk_id in map_num_frames_to_spk_id[num_frames]:
                spk_data[spk_id]["audio"] = spk_audio
        
        det_time = time.time() - start_time
        #print('det after extracting audio features: ', det_time)
        
        for num_frames in map_num_frames_to_spk_id:
            with torch.no_grad():
                faceFeature = torch.FloatTensor(np.array([spk_data[spk_id]["face_crop"] for spk_id in map_num_frames_to_spk_id[num_frames]])).to(device)
                audioFeature = torch.FloatTensor(np.array([spk_data[spk_id]["audio"] for spk_id in map_num_frames_to_spk_id[num_frames]])).to(device)
                bodyFeature = torch.FloatTensor(np.array([spk_data[spk_id]["kp_central"] for spk_id in map_num_frames_to_spk_id[num_frames]])).to(device)

                if bodyPose:
                    bodyFeature = torch.FloatTensor(np.array([spk_data[spk_id]["kp_central"] for spk_id in map_num_frames_to_spk_id[num_frames]])).to(device)
                    outsFAB, _, _, _= asd_model.model(faceFeature, audioFeature, bodyFeature[:, :, : 11, :] if upperBody else bodyFeature)
                    out_score = asd_model.lossFAB.module.FC(outsFAB.squeeze(1)) #[-1]
                else:
                    outsFA, _, _ = asd_model.model(faceFeature, audioFeature, None)
                    out_score = asd_model.lossFA.module.FC(outsFA.squeeze(1)) #[-1]
                predScore = F.softmax(out_score, dim = -1)[:, 1]
                for idx, spk_id in enumerate(map_num_frames_to_spk_id[num_frames]):
                    spk_data[spk_id]["speaking"] = predScore[idx].detach().cpu().numpy()
        
        det_time = time.time() - start_time
        #print('det after processing: ', det_time)

        img_show = frame.copy()

        #img_show = draw_skeleton(img_show,
        #                         keypoints,
        #                         scores,
        #                         openpose_skeleton=openpose_skeleton,
        #                         kpt_thr=0.25)
        
        for spk_id in speakers_in_last_frame:
            num_frames = len(spk_data[spk_id]["face_bbox"])
            if num_frames > 0:
                cv2.rectangle(img_show, (spk_data[spk_id]["face_bbox"][-1][0], spk_data[spk_id]["face_bbox"][-1][1]), (spk_data[spk_id]["face_bbox"][-1][2], spk_data[spk_id]["face_bbox"][-1][3]), (0, 255 if spk_data[spk_id]["speaking"] >= speech_activity_acceptance_thr else 0, 0 if spk_data[spk_id]["speaking"] >= speech_activity_acceptance_thr else 255), 2)

        img_show = cv2.resize(img_show, (1440, 1080))
        cv2.imshow('img', img_show)
        cv2.waitKey(1)

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    audio_stream.stop()
    audio_stream.close()        


def main():
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description = "Model Training")
    # Training parameters
    parser.add_argument('--mode',              nargs="?",  default="test",      choices=["train", "test", "demo"], help='Mode of execution')
    parser.add_argument('--lr',                type=float, default=0.001,       help='Learning rate')  # Original LightASD: 0.001
    parser.add_argument('--lrDecay',           type=float, default=0.95,        help='Learning rate decay rate')  # Original LightASD: 0.95
    parser.add_argument('--maxEpoch',          type=int,   default=60,          help='Maximum number of epochs')  # Original LightASD: 30
    parser.add_argument('--testInterval',      type=int,   default=1,           help='Test and save every [testInterval] epochs')
    parser.add_argument('--batchSize',         type=int,   default=2000,        help='Dynamic batch size, default is 2000 frames')  # Original LightASD: 2000 (for AVA-ActiveSpeaker)
    parser.add_argument('--nDataLoaderThread', type=int,   default=64,          help='Number of loader threads')  # Original LightASD: 64
    parser.add_argument('--optim',             type=str,   default="Adam",      help='Optimiser type')  # Original LightASD: Adam
    parser.add_argument('--lossScheduling',    type=str,   default="epochStep", help='Loss decay scheduling approach')  # Original LightASD: epochStep
    parser.add_argument('--numWarmupEpochs',   type=float, default=0.,          help='Number of warmup epochs; a non-integer number implies stopping warmup mid-epoch')  # Original LightASD: 0
    parser.add_argument('--initialTemp',       type=float, default=1.3,         help='Initial temperature')  # Original LightASD: 1.3
    parser.add_argument('--tempDecayType',     type=str,   default="linear",    help='Type of decay of the temperature')  # Original LightASD: linear
    parser.add_argument('--tempDecayRate',     type=float, default=0.02,        help='Rate of decay of the temperature')  # Original LightASD: 0.02
    
    # Model architecture
    parser.add_argument('--sizeVideoInput', type=int, default=112, help='Length of one of the dimensions of the video input')  # Original LightASD: 112
    parser.add_argument('--bodyPose', action='store_true', help='Model uses body pose information')  # Original LightASD: False
    parser.add_argument('--bodyPoseAugProb', type=float, default=0., help='Maximum range of fluctuation of a pose keypoint position')  # Original LightASD: N/A
    parser.add_argument('--bodyPoseAugMag', type=float, default=0., help='Maximum range of fluctuation of a pose keypoint position')  # Original LightASD: N/A
    parser.add_argument('--upperBody', action='store_true', help='Pose information includes only the keypoints of the upper body')  # Original LightASD: N/A
    
    # Data path
    parser.add_argument('--dataPathWASD',  type=str, default="WASD", help='Save path of WASD dataset')
    parser.add_argument('--baseSavePath',  type=str, default="exps")
    parser.add_argument('--id',  type=str, default="noid", help='An identifier to be appended to the name of the experiment result folder for the sake of differentiation')
    args = parser.parse_args()    
    
    experimentCanonicalSavePath = f"exp_{args.lr}_{args.lrDecay}_{args.maxEpoch}_{args.batchSize}_{args.optim}_{args.lossScheduling}_{args.numWarmupEpochs}_{args.initialTemp}_{args.tempDecayType}_{args.tempDecayRate}_{args.sizeVideoInput}_"
    experimentCanonicalSavePath += ("T" if args.bodyPose else "F")
    if args.bodyPose:
        experimentCanonicalSavePath += f"_{args.bodyPoseAugProb}_{args.bodyPoseAugMag}"
        experimentCanonicalSavePath += "_" + ("upper" if args.upperBody else "whole")
    experimentCanonicalSavePath += f"_{args.id}_config"
    args.savePath = os.path.join(args.baseSavePath, experimentCanonicalSavePath)
    
    args.modelSavePath      = os.path.join(args.savePath,      'model')
    args.scoreSavePath      = os.path.join(args.savePath,      'score.txt')
    args.trialPathWASD      = os.path.join(args.dataPathWASD,  'csv')
    args.faceOrigPathWASD   = os.path.join(args.dataPathWASD,  'orig_videos')
    args.audioOrigPathWASD  = os.path.join(args.dataPathWASD,  'orig_audios')
    args.facePathWASD       = os.path.join(args.dataPathWASD,  'clips_videos')
    args.audioPathWASD      = os.path.join(args.dataPathWASD,  'clips_audios')
    args.trainTrialWASD     = os.path.join(args.trialPathWASD, 'train_loader_body.csv')
    args.trainOrigKpBody    = os.path.join(args.trialPathWASD, 'train_orig_kp_body.csv')

    args.evalTrialWASD  =  os.path.join(args.trialPathWASD, 'val_loader_body.csv')
    args.evalOrig       =  os.path.join(args.trialPathWASD, 'val_orig_body.csv')
    args.evalOrigKpBody =  os.path.join(args.trialPathWASD, 'val_orig_kp_body.csv')
    args.evalCsvSave    =  os.path.join(args.savePath,      'val_res.csv')
    args.bestEvalCsvSave = os.path.join(args.savePath,      'best_val_res.csv')
    
    os.makedirs(args.modelSavePath, exist_ok = True)
    os.makedirs(args.dataPathWASD, exist_ok = True)
    
    g = torch.Generator()
    g.manual_seed(42)
    
    if args.mode == "train":
        loader = train_loader(trialFileName     = args.trainTrialWASD, \
                              keypointsFileName = args.trainOrigKpBody, \
                              facePath        = os.path.join(args.facePathWASD, 'train'), \
                              audioPath         = os.path.join(args.audioPathWASD, 'train'), \
                              **vars(args))
        trainLoader = torch.utils.data.DataLoader(loader, batch_size = 1, shuffle = True, num_workers = args.nDataLoaderThread, pin_memory = True, worker_init_fn = seed_worker, generator = g)
    
    if args.mode != "demo":
        loader = val_loader(trialFileName     = args.evalTrialWASD, \
                            keypointsFileName = args.evalOrigKpBody, \
                            facePath        = os.path.join(args.facePathWASD, 'val'), \
                            audioPath         = os.path.join(args.audioPathWASD , 'val'), \
                            **vars(args))
        valLoader = torch.utils.data.DataLoader(loader, batch_size = 1, shuffle = False, num_workers = args.nDataLoaderThread, pin_memory = True, worker_init_fn = seed_worker, generator = g)
    
    if args.mode == "train":
        model_files = glob.glob('%s/last_0*.model'%args.modelSavePath)
    else:
        model_files = glob.glob('%s/best_0*.model'%args.modelSavePath)
        print(model_files)
    
    mAPs = []
    if len(model_files) >= 1:
        mAPs = pickle.load(open('%s/maps.pkl'%args.modelSavePath, 'rb'))
        print("Model loaded from previous state!")
        epoch = int(os.path.splitext(os.path.basename(model_files[-1]))[0][5:]) + 1
        s = ASD(epoch = epoch, **vars(args))
        s.loadParameters(model_files[-1])
    else:
        epoch = 1
        s = ASD(epoch = epoch, **vars(args))
    
    if args.mode == "demo":
        execute_demo(s, **vars(args))        
    else:
        categories = ["Interview", "Debate", "Podcast", "React", "Police", "Overall"]
        
        if args.mode == "test":
            curr_mAPs = s.evaluate_network(epoch = epoch, loader = valLoader, **vars(args))
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, %s mAP %2.2f%%, %s mAP %2.2f%%, %s mAP %2.2f%%, %s mAP %2.2f%%, %s mAP %2.2f%%, %s mAP %2.2f%%"%(epoch, categories[0], curr_mAPs[categories[0]], categories[1], curr_mAPs[categories[1]], categories[2], curr_mAPs[categories[2]], categories[3], curr_mAPs[categories[3]], categories[4], curr_mAPs[categories[4]], categories[5], curr_mAPs[categories[5]]))
        else:
            scoreFile = open(args.scoreSavePath, "a+")
           
            while True:        
                loss, lr = s.train_network(epoch = epoch, loader = trainLoader, **vars(args))
                torch.cuda.empty_cache()
                
                if epoch % args.testInterval == 0:
                    s.saveParameters(args.modelSavePath + "/last_%04d.model"%epoch)
                    removable_previous_models = [x for x in glob.glob('%s/last_*.model'%args.modelSavePath) if x != '%s/last_%04d.model'%(args.modelSavePath, epoch)]
                    for filepath in removable_previous_models:
                        os.remove(filepath)
                    
                    mAPs.append(s.evaluate_network(epoch = epoch, loader = valLoader, **vars(args)))
                    pickle.dump(mAPs, open('%s/maps.pkl'%args.modelSavePath, 'wb'), protocol = pickle.HIGHEST_PROTOCOL)
                    print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, %s mAP %2.2f%%, %s mAP %2.2f%%, %s mAP %2.2f%%, %s mAP %2.2f%%, %s mAP %2.2f%%, %s mAP %2.2f%%, bestmAP %2.2f%%"%(epoch, categories[0], mAPs[-1][categories[0]], categories[1], mAPs[-1][categories[1]], categories[2], mAPs[-1][categories[2]], categories[3], mAPs[-1][categories[3]], categories[4], mAPs[-1][categories[4]], categories[5], mAPs[-1][categories[5]], max([x[categories[5]] for x in mAPs])))
                    scoreFile.write("%d epoch, LR %f, LOSS %f, %s mAP %2.2f%%, %s mAP %2.2f%%, %s mAP %2.2f%%, %s mAP %2.2f%%, %s mAP %2.2f%%, %s mAP %2.2f%%, bestmAP %2.2f%%\n"%(epoch, lr, loss, categories[0], mAPs[-1][categories[0]], categories[1], mAPs[-1][categories[1]], categories[2], mAPs[-1][categories[2]], categories[3], mAPs[-1][categories[3]], categories[4], mAPs[-1][categories[4]], categories[5], mAPs[-1][categories[5]], max([x[categories[5]] for x in mAPs])))
                    scoreFile.flush()
                    if mAPs[-1][categories[5]] == max([x[categories[5]] for x in mAPs]):
                        s.saveParameters(args.modelSavePath + "/best_%04d.model"%epoch)
                        removable_previous_models = [x for x in glob.glob('%s/best_*.model'%args.modelSavePath) if x != '%s/best_%04d.model'%(args.modelSavePath, epoch)]#
                        for filepath in removable_previous_models:
                            os.remove(filepath)
                        shutil.copyfile(args.evalCsvSave, args.bestEvalCsvSave)
                    torch.cuda.empty_cache()

                if epoch >= args.maxEpoch:
                    quit()

                epoch += 1

if __name__ == '__main__':
    seed_everything(42)
    main()
