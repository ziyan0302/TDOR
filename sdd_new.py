# This script should be put at cvpr2022/data/SDD/sdd_new.py
# and this script can be tested at cvpr2022 folder after adding __init__.py in both data folder and SDD folder
# test command is to stay at cvpr2022 folder: python3 -m data.SDD.sdd_new
import os
import numpy as np
import cv2
# from cvpr2022.data.trajDataset import TrajDataset,tsfm,normalize_imagenet,load_dict
import pdb
from ..trajDataset import TrajDataset,tsfm,normalize_imagenet,load_dict
import yaml
import copy
import matplotlib.pyplot as plt

scene_names = ['bookstore', 'coupa', 'deathCircle', 'gates', 'hyang', 'little', 'nexus', 'quad']

# sdd_root = "/home/ziyan/datasets/Y-net_data/SDD"
sdd_root = "/home/ziyan/TDOR/cvpr2022/data/SDD"

with open(os.path.join(sdd_root, 'estimated_scales.yaml'), 'r') as hf:
    scales_yaml_content = yaml.load(hf, Loader=yaml.FullLoader)


class sdd_new(TrajDataset):

    def __init__(self,dataset="trajnet", type="train",hist_len=8, fut_len=12, horizon=20,grid_extent=20):
        super().__init__(horizon)

        self.d_s = 12  # down sampling rate of sequences for trajectories

        self.t_h = (hist_len-1) * self.d_s  # length of track history
        self.t_f = fut_len * self.d_s  # length of predicted trajectory
        self.grid_extent = grid_extent  # The extent of the cropped scene around the agent (l, r, d, u)

        # dataroot="./cvpr2022/data/SDD/"+dataset+"_"+type+".pkl"
        dataroot="/home/ziyan/TDOR/cvpr2022/data/SDD/"+dataset+"_"+type+".pkl"
        self.F, self.T,frame_file  = load_dict(dataroot) #F: frame_dict T: track_dict

        self.scales=[]
        self.D=[]
        self.images=[[],[],[],[]] 
        # [0]: original; [1]:90 counterclock; [2]: 180 counterclock; [3]: 270 counterclock
        # [4]: flip from [0]; [5]: flip from [1]; [6]: flip from [2]; [7]: flip from [3];
        self.Ts=[self.T,[],[],[]]
        # T[i] relates to images[i] # when images have new augmentation, T would have corresponding aug too.

        self.load_maps()
        if (0):
            len(self.images[0])
        if type=="train":
            self.augment_data()
        ## debug2
        if (0):
            len(self.images)
            image = self.images[0]
            images = self.images
            scales = self.scales
            padsize = self.img_size//2
            interpolation=cv2.INTER_LINEAR
            for i_img in range(len(image)):
                i_img = 1
                factor=scales[i_img]*100/grid_extent
                img = cv2.resize(image[i_img], (0, 0), fx=factor, fy=factor,interpolation=interpolation)
                img1 = cv2.copyMakeBorder(img, padsize, padsize, padsize, padsize, cv2.BORDER_CONSTANT, value=0)
                img1.shape
                img.shape
                image[i_img].shape
                image[i_img].shape[0]*factor
                import matplotlib.pyplot as plt
                plt.subplot(311)
                plt.imshow(image[i_img])
                plt.subplot(312)
                plt.imshow(img)
                plt.subplot(313)
                plt.imshow(img1)
                plt.show()
                images[i][i_img] = img

        self.resize_pad(self.images,self.scales,self.img_size//2,self.grid_extent)

        for i in range(len(frame_file)):
            ds_id=int(frame_file[i][0])
            k=frame_file[i][1]
            frame=frame_file[i][2]
            if type == "train":
                for m in range(len(self.Ts)):
                    self.D.append((m, ds_id, k, frame))
            else:
                self.D.append((0, ds_id, k, frame))



        # if dataset=="trajnet":
        #     for key in range(len(self.T)):
        #         if type == "train":
        #             if key not in [7,8,18,25,26,33,38,40,41,42,43,51,52,56,57,58,59]:
        #
        #                 track = self.T[key]
        #
        #                 for key1 in track.keys():
        #
        #                     cur_frame = track[key1][self.t_h][0]
        #
        #                     for m in range(len(self.Ts)):
        #                         self.D.append((m, key, key1, cur_frame))
        #
        #         else:
        #             if key in [7,8,18,25,26,33,38,40,41,42,43,51,52,56,57,58,59]:
        #                 track = self.T[key]
        #
        #                 for key1 in track.keys():
        #                     cur_frame = track[key1][self.t_h][0]
        #
        #                     self.D.append((0, key, key1, cur_frame))
        # else:
        #     frame_file=np.load("./data/SDD/"+type+".npy").astype(int)
        #
        #     for i in range(len(frame_file)):
        #         if type == "train":
        #             for m in range(len(self.Ts)):
        #                 self.D.append([m,frame_file[i][0]-1,frame_file[i][1],frame_file[i][2]])
        #         else:
        #             self.D.append([0,frame_file[i][0]-1, frame_file[i][1], frame_file[i][2]])


    def __len__(self):
        return len(self.D)

    def load_maps(self):
        for i, scene_name in enumerate(scene_names):
            scene_root = os.path.join(sdd_root, scene_name)

            s = os.listdir(scene_root)

            s.sort()
            for scene_video_id in s:
                scale = scales_yaml_content[scene_name][scene_video_id]['scale']

                file_path = os.path.join(scene_root, scene_video_id, 'reference.jpg')
                img = cv2.imread(file_path).astype(np.float32)/255

                self.images[0].append(img) # ref jpg from every scene folder
                self.scales.append(scale) 

    def augment_data(self):
        for i in range(len(self.T)):
            for k in [1,2,3]:
                T = copy.deepcopy(self.Ts[k-1][i])

                image=self.images[k-1][i]

                image_rot = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

                self.images[k].append(image_rot)

                y0= image_rot.shape[0]  # shape after rotate

                for j in T.keys():
                    m = copy.deepcopy(T[j][:,1])
                    T[j][:,1] = T[j][:,2]
                    T[j][:,2] = y0-m

                self.Ts[k].append(T)

        for k in range(4):
            self.images.append([])
            self.Ts.append([])
            T=self.Ts[k]
            for i in range(len(T)):
                image = self.images[k][i]
                x0= image.shape[1]

                flip_image= cv2.flip(image, 1)

                self.images[4+k].append(flip_image)

                T_flip = copy.deepcopy(T[i])

                for j in T_flip.keys():
                    T_flip[j][:,1] = x0-T_flip[j][:,1]
                self.Ts[4+k].append(T_flip)

    def get_img(self, img, ref_pos):

        x0 = round(ref_pos[0]* self.img_center/self.grid_extent)#
        y0 = round(ref_pos[1]* self.img_center/self.grid_extent)#

        img_cropped = img[y0: y0 + self.img_size, x0: x0 + self.img_size]

        return img_cropped

        

    def __getitem__(self, idx):

        m,ds_id,t_id,cur_frame=self.D[idx]

        full_track = self.Ts[m][ds_id][t_id]

        stpt=np.argwhere(full_track[:, 0] == cur_frame).item() - self.t_h

        scale=self.scales[ds_id]

        track=full_track[stpt:,1:3]*scale

        ## debug3
        if (0):
            len(full_track) # 406
            len(track) # 390 # stpt = 16, curr_frame = no.100
            ref_pos = track[self.t_h] # curr location
            import matplotlib.pyplot as plt
            len(self.Ts[1][0][0])
            track1 = self.Ts[1][0][0][:,1:]
            track1 = track1*scale* self.img_center/self.grid_extent
            track1[:,0] = track1[:,0]+self.img_size/2
            track1[:,1] = track1[:,1]+self.img_size/2

            len(track1)
            ref_pos in track1
            x0 = round(ref_pos[0]* self.img_center/self.grid_extent)#
            y0 = round(ref_pos[1]* self.img_center/self.grid_extent)#
            img_cropped_tdor = self.images[1][0][y0: y0 + self.img_size, x0: x0 + self.img_size]
            
            target_pos = track1[self.t_h]
            targethist = track1[self.d_s:self.t_h + self.d_s+1:self.d_s] - track1[self.t_h] + [16,16]
            targetfut = track1[self.t_h :self.t_h + self.t_f + 1:self.d_s] - track1[self.t_h] + [16,16]
            img_cropped = self.images[1][0][round(target_pos[1]-16): round(target_pos[1]+16), round(target_pos[0]-16): round(target_pos[0]+16)]
            img_cropped.shape
            print(ref_pos)
            plt.subplot(131)
            plt.imshow(self.images[1][0])
            plt.plot(*zip(*(track1)),linestyle='--', marker='.', color='b')
            plt.plot(*zip(*(track1[self.d_s:self.t_h + 1:self.d_s])),linestyle='--', marker='.', color='y')
            plt.plot(*zip(*(track1[self.t_h + self.d_s:self.t_h + self.t_f + 1:self.d_s])),linestyle='--', marker='.', color='black')
            plt.scatter(x0,y0,color='r')
            plt.scatter(x0+100,y0+100,color='g')
            plt.scatter(target_pos[0],target_pos[1],color='r')
            plt.subplot(132)
            plt.imshow(img_cropped_tdor)
            plt.scatter(100,100, color='g')
            plt.subplot(133)
            plt.imshow(img_cropped)
            plt.plot(*zip(*(targethist)), color='y')
            plt.plot(*zip(*(targetfut)), color='black')
            plt.scatter(16,16,color='g')
            # plt.plot(*zip(*(track1[self.d_s:self.t_h + self.t_f ] - target_pos + [16,16])), color='b')
            plt.show()
        
        if (1):
            import matplotlib.pyplot as plt
            imgs_list = []
            track1=full_track[stpt:,1:3]*scale*self.img_center/self.grid_extent + [self.img_size/2,self.img_size/2]
            target_pos = track1[self.t_h]
            crop_size = 16
            # targethist = track1[self.d_s:self.t_h + self.d_s+1:self.d_s] - target_pos + [crop_size,crop_size]
            # targetfut = track1[self.t_h :self.t_h + self.t_f + 1:self.d_s] - target_pos + [crop_size,crop_size]
            targettrack = track1[self.d_s :self.t_h + self.t_f + 1:self.d_s] 
            
            # for i_crop in range(1,targettrack.shape[0]):
            for i_crop in range(1,targettrack.shape[0]):
                target_pos = targettrack[i_crop]
                img_cropped_small = self.images[m][ds_id][round(target_pos[1]-crop_size): round(target_pos[1]+crop_size),\
                     round(target_pos[0]-crop_size): round(target_pos[0]+crop_size)]
                
                fig, ax = plt.subplots(figsize=(4,4), dpi=4)
                plt.imshow(img_cropped_small)
                plt.plot(*zip(*(targettrack[i_crop-1:i_crop+1]-target_pos+[crop_size,crop_size])), color='b')
                # plt.scatter(16,16,color='g')
                # plt.show()

                fig.tight_layout(pad=0)
                plt.axis('off')
                plt.subplots_adjust(0,0,1,1,0,0)
                ax.margins(0)
                fig.canvas.draw()
                image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                image_from_plot.shape
                plt.figure()
                plt.imshow(image_from_plot)
                # plt.show()
                imgs_list.append(image_from_plot)
                plt.close('all')


        ref_pos = track[self.t_h]

        start_pos = track[0]

        delta = ref_pos - start_pos

        theta = np.degrees(np.arctan2(delta[1], delta[0]))

        rel_track = track - ref_pos # set origin at curr loc

        vel_hist = rel_track[self.d_s:self.t_h + 1:self.d_s] - rel_track[ 0:self.t_h + 1 - self.d_s:self.d_s]  # using velocity

        fut = rel_track[self.t_h + self.d_s:self.t_h + self.t_f + 1:self.d_s]

        img_vis = self.get_img(self.images[m][ds_id], ref_pos)
        
        

        img = normalize_imagenet(tsfm.ToTensor()(img_vis))

        fut_indefinite = rel_track[self.t_h:]

        waypts_e, waypt_lengths_e, bc_targets = self.get_expert_waypoints(fut_indefinite/self.grid_extent)

        r_mat = self.rmat(theta)

        neighbors = self.get_nei(self.F,self.d_s,self.Ts,cur_frame - self.t_h, m, ds_id, t_id, track, scale, r_mat)#self.get_nei(cur_frame-self.t_h,m,ds_id,t_id,track,scale,r_mat)

      #  vel_hist=rel_track[0:self.t_h + 1:self.d_s]

        return vel_hist, neighbors, fut, img, waypts_e, bc_targets, waypt_lengths_e,  r_mat,scale ,ref_pos, ds_id, imgs_list


if __name__ == '__main__':

    tr_set = sdd_new("sdd", horizon=20, fut_len=12, grid_extent=20)
    vel_hist, neighbors, fut, img, waypts_e, bc_targets, waypt_lengths_e,  r_mat,scale ,ref_pos, ds_id, imgs_list= tr_set.__getitem__(1)
    pdb.set_trace()
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(img.numpy().transpose(1,2,0))
    # plt.show()

    # define subplot grid
    fig, axs = plt.subplots(nrows=6, ncols=3, figsize=(5, 5))
    plt.subplots_adjust(hspace=0.5)

    # loop through tickers and axes
    for i_iter, (img, ax) in enumerate(zip(imgs_list, axs.ravel())):
        # filter df for ticker and plot on specified axes

        ax.imshow(img)

        # chart formatting
        ax.set_title(str(i_iter))
        ax.set_xlabel("")
    pdb.set_trace()

    plt.show()