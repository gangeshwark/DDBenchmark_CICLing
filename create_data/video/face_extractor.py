import cPickle
import glob
import os
import sys

import imageio
import openface


def extract_frames(files, modelfile, sampled_fps=5, sampled_frame_idx=None, output_dim=96, save_as_jpg=False,
                   output_loc=None):
    # files = "/home/nlp-cic/deception_multimodal/data/Clips/deceptive/*.mp4"
    # output_loc="/home/nlp-cic/deception_multimodal/data/Clips/deceptive_sampled"
    # sampled_frame_idx=10
    # sampled_fps=5
    # output_dim=96

    model = openface.AlignDlib(modelfile)  # open model file for face extraction
    if save_as_jpg and not os.path.exists(output_loc):
        os.makedirs(output_loc)

    videos = []
    all_files = glob.glob(files)  # get the list of all the .mp4 files
    video_count = len(all_files)
    for vid_num, filepath in enumerate(all_files):
        video = imageio.get_reader(filepath, 'ffmpeg')  # open video file
        filename = filepath[len(filepath) - filepath[::-1].find('/'):]

        if save_as_jpg:
            extract_loc = output_loc + "/" + filename
            if not os.path.exists(extract_loc):
                os.makedirs(extract_loc)
            else:
                continue

        meta_data = video.get_meta_data()
        print "Processing file [%d/%d]: %s" % (vid_num + 1, video_count, filename)
        print meta_data

        if sampled_frame_idx == None:
            sampled_frame_idx = int(meta_data['fps'] / sampled_fps)
        nframes = meta_data['nframes']

        frames = []

        frame_idxs = xrange(sampled_frame_idx - 1, nframes, sampled_frame_idx)
        lenfs = len(frame_idxs)
        for num, idx in enumerate(frame_idxs):

            sys.stdout.write('  Processed frames [%d/%d]\r' % (num, lenfs))
            sys.stdout.flush()
            try:
                frame = video.get_data(idx)
            except Exception:
                continue
            bb = model.getLargestFaceBoundingBox(frame)  # get boundingbox co-ordinates of the face
            alignedFace = model.align(output_dim, frame, bb,
                                      landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)  # crop the face as (output_dim x output_dim) image

            if alignedFace != None:
                frames.append(alignedFace)
                if save_as_jpg:
                    output_file = extract_loc + "/" + str(num) + ".jpg"
                    imageio.imwrite(output_file, alignedFace)
        print '  Processed frames [%d/%d]\n' % (lenfs, lenfs)
        videos.append(frames)
    return videos


if __name__ == '__main__':
    deceptive_videos = extract_frames(files="/home/navonil/deception_detection/Clips/deceptive/*.mp4",
                                      modelfile="/home/gangeshwark/openface/models/dlib/shape_predictor_68_face_landmarks.dat",
                                      #                                    sampled_fps=5,
                                      sampled_frame_idx=10,
                                      output_dim=96
                                      )
    truthful_videos = extract_frames(files="/home/navonil/deception_detection/Clips/truthful/*.mp4",
                                     modelfile="/home/gangeshwark/openface/models/dlib/shape_predictor_68_face_landmarks.dat",
                                     #                                    sampled_fps=5,
                                     sampled_frame_idx=10,
                                     output_dim=96
                                     )
    cPickle.dump([deceptive_videos, truthful_videos], open("deception_fixed_frames.p", 'wb'))
