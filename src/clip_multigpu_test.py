"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license

This code was based on the file run_cnaps.py (https://github.com/cambridge-mlg/cnaps/blob/master/src/run_cnaps.py)
from the cambridge-mlg/cnaps library (https://github.com/cambridge-mlg/cnaps).

The original license is included below:

Copyright (c) 2019 John Bronskill, Jonathan Gordon, and James Requeima.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. 
"""

import os
import time
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn

from data.dataloaders import DataLoader
from models.few_shot_recognisers import MultiStepFewShotRecogniser, FullRecogniser
from utils.args import parse_args
from utils.ops_counter import OpsCounter
from utils.optim import cross_entropy, init_optimizer
from utils.data import get_clip_loader, unpack_task, attach_frame_history
from utils.logging import print_and_log, get_log_files, stats_to_str
from utils.eval_metrics import TrainEvaluator, ValidationEvaluator, TestEvaluator

from models.classifiers import CLIPLinearClassifier, LinearClassifier, VersaClassifier, PrototypicalClassifier, MahalanobisClassifier, CLIPPromptClassifier
from models.poolers import MeanPooler

import torch.nn as nn
import gc 

#torch.multiprocessing.set_sharing_strategy('file_system')
import clip
def main():
    learner = Learner()
    learner.run()



class TextCLIP(nn.Module):
    def __init__(self, model, device=None) :
        super(TextCLIP, self).__init__()
        self.model = model
        self.convert_to_fp32()
        self.device = device
        self.model.to(self.device)
        
    def forward(self,text):
        #print(text.device)
        #print(self.device)
        return self.model.encode_text(text)
    
    def convert_to_fp32(self):
        for p in self.model.parameters():
            p.data = p.data.float()
    

class ImageCLIP(nn.Module):
    def __init__(self, model) :
        super(ImageCLIP, self).__init__()
        self.model = model
        self.convert_to_fp32()
        
    def forward(self,image):
        return self.model.encode_image(image)

    def convert_to_fp32(self):
        for p in self.model.parameters():
            p.data = p.data.float()


class Learner:
    def __init__(self):
        self.args = parse_args(learner='gradient-learner')

        self.checkpoint_dir, self.logfile, self.checkpoint_path_validation, self.checkpoint_path_final \
            = get_log_files(self.args.checkpoint_dir, self.args.model_path)

        print_and_log(self.logfile, "Options: %s\n" % self.args)
        print_and_log(self.logfile, "Checkpoint Directory: %s\n" % self.checkpoint_dir)

        self.batch_size = self.args.batch_size

        self.device = torch.device("cuda:0")
        
        self.init_dataset()
        self.init_evaluators()
        self.model_text, self.model_image = self.init_model()
        self.loss = cross_entropy

        self.output_size = 512
        
        classifier = self.args.classifier
        
        if classifier == 'linear':
            # classifier head will instead be appended per-task during train/test
            self.classifier = LinearClassifier(self.output_size)
        elif classifier == "sklinear":
            self.classifier =  SKLinearClassifier(self.output_size)
        elif classifier == 'versa':
            self.classifier = VersaClassifier(self.output_size)
        elif classifier == 'proto':
            self.classifier = PrototypicalClassifier()
        elif classifier == 'mahalanobis':
            self.classifier = MahalanobisClassifier()
        elif classifier == "clip":
            self.classifier = CLIPLinearClassifier(self.output_size, self.model_text)
        elif classifier == "coop":
            self.classifier = CLIPPromptClassifier(self.output_size, self.model_text, classifier)
        elif classifier == "cocoop":
            self.classifier = CLIPPromptClassifier(self.output_size, self.model_text, classifier) 
       
        self.classifier._set_device(self.device)
        self.pooler = MeanPooler(T=self.args.clip_length)
        #self.train_task_fn = self.train_task_in_batches if self.args.with_lite else self.train_task

        #self.init_trainer()


    def init_dataset(self):

        dataset_info = {
            'mode': self.args.mode,
            'data_path': self.args.data_path,
            'train_object_cap': self.args.train_object_cap,
            'with_train_shot_caps': self.args.with_train_shot_caps,
            'with_cluster_labels': False,
            'train_way_method': self.args.train_way_method,
            'test_way_method': self.args.test_way_method,
            'train_shot_methods': [self.args.train_context_shot_method, self.args.train_target_shot_method],
            'test_shot_methods': [self.args.test_context_shot_method, self.args.test_target_shot_method],
            'train_tasks_per_user': self.args.train_tasks_per_user,
            'test_tasks_per_user': self.args.test_tasks_per_user,
            'train_task_type' : self.args.train_task_type,
            'test_set': self.args.test_set,
            'shots': [self.args.context_shot, self.args.target_shot],
            'video_types': [self.args.context_video_type, self.args.target_video_type],
            'clip_length': self.args.clip_length,
            'train_num_clips': [self.args.train_context_num_clips, self.args.train_target_num_clips],
            'test_num_clips': [self.args.test_context_num_clips, self.args.test_target_num_clips],
            'subsample_factor': self.args.subsample_factor,
            'frame_size': self.args.frame_size,
            'annotations_to_load': self.args.annotations_to_load,
            'preload_clips': self.args.preload_clips,
            'num_workers': self.args.num_workers
        }

        dataloader = DataLoader(dataset_info)
        self.train_queue = dataloader.get_train_queue()
        self.validation_queue = dataloader.get_validation_queue()
        self.test_queue = dataloader.get_test_queue()
        
    def init_model(self):

        model, preprocess = clip.load('ViT-B/32', jit=False)
        
        model_text = TextCLIP(model, self.device)
        model_image = ImageCLIP(model)

        #model_text = nn.DataParallel(model_text)
        #model_image = nn.DataParallel(model_image)

        model_text.to(self.device)
        model_image.to(self.device)

        return model_text, model_image

    """
    def init_task_model(self):
        #return self.model
        model = self.init_model()
        model.load_state_dict(self.model.state_dict(), strict=False)
        self.zero_grads(model)
        return model
    """


    def zero_grads(self, model):
        # init grad buffers to 0, otherwise None until first backward
        for param in model.parameters():
            if param.requires_grad:
                param.grad = param.new(param.size()).fill_(0)
       
    def copy_grads(self, src_model, dest_model):
        for (src_param_name, src_param), (dest_param_name, dest_param) in zip(src_model.named_parameters(), dest_model.named_parameters()):
            assert src_param_name == dest_param_name
            if dest_param.requires_grad:
                dest_param.grad += src_param.grad.detach()
                dest_param.grad.clamp_(-10, 10)
    
    def init_evaluators(self):
        self.train_metrics = ['frame_acc']
        self.evaluation_metrics = ['frame_acc', 'frames_to_recognition', 'video_acc']
        self.train_evaluator = TrainEvaluator(self.train_metrics)
        self.validation_evaluator = ValidationEvaluator(self.evaluation_metrics)
        self.test_evaluator = TestEvaluator(self.evaluation_metrics, self.checkpoint_dir)

    def run(self):

        #torch.save(self.model.state_dict(), self.checkpoint_path_final)

        self.test(self.checkpoint_path_final)
        pass


    def eval_test(self, target_frames_by_video, target_paths_by_video, target_labels_by_video):
        with torch.no_grad():
            for video_frames, video_paths, video_label in zip(target_frames_by_video, target_paths_by_video, target_labels_by_video):
                video_clips = attach_frame_history(video_frames, self.args.clip_length)
                #print(self.args.clip_length)
                #print(video_clips.shape)
                features = []
                for i in range(0, video_clips.shape[0], self.args.batch_size):
                    bc = video_clips[i:i+self.args.batch_size]
                    bc = bc.to(self.device)
                    sz = bc.size()
                    bc = bc.view(-1, sz[-3], sz[-2], sz[-1]) if bc.dim() >= 5 else bc

                    features.append(self.model_image(bc))
                
                del video_clips
                #vid_logits = [self.model_image(video_clips[i:i+self.args.batch_size]) for i in range(0, video_clips.shape[0], self.args.batch_size)]
                
                """
                clip_loader = get_clip_loader(video_clips, self.batch_size)
                #clip_loader = get_clip_loader(video_frames, self.batch_size)

                features = []

                for batch_clips in clip_loader:

                    sz = batch_clips.size()
                    
                    #batch_clips = batch_clips.to(torch.device("cuda:0"))
                    batch_clips = batch_clips.to(self.device)
                    batch_clips = batch_clips.view(-1, sz[-3], sz[-2], sz[-1]) if batch_clips.dim() >= 5 else batch_clips
                    batch_features = self.model_image(batch_clips)

                    #gc.collect()
                    
                    #features.append(batch_features)
                """
        pass



    def test(self, path):

        self.model_text, self.model_image = self.init_model()
        
        # loop through test tasks (num_test_users * num_test_tasks_per_user)
        num_test_tasks = self.test_queue.num_users * self.args.test_tasks_per_user
        print(self.args.test_tasks_per_user)
        for step, task_dict in enumerate(self.test_queue.get_tasks()):
            
            context_clips, context_paths, context_labels, target_frames_by_video, target_paths_by_video, target_labels_by_video, object_list = unpack_task(task_dict, torch.device("cpu"), context_to_device=False, preload_clips=self.args.preload_clips)
           
            print(object_list)
            #print(context_paths)
            # if this is a user's first task, cache their target videos (as they remain constant for all their tasks - ie. num_test_tasks_per_user)
            if step % self.args.test_tasks_per_user == 0:
                cached_target_frames_by_video, cached_target_paths_by_video, cached_target_labels_by_video = target_frames_by_video, target_paths_by_video, target_labels_by_video


            # Initialize Current Model
            #model.set_test_mode(True)

            # take a few grad steps using context clips
            t1 = time.time()
            learning_args=(self.args.learning_rate, self.loss, 'sgd', 0.1)
            
            # Train 
            #model = self.trainer.run(model, context_clips, context_labels, learning_args, ops_counter=self.ops_counter)
            context_clip_loader = get_clip_loader(context_clips, self.batch_size, with_labels=True)

            task_embedding = None 

            with torch.no_grad():
                features = []

                for batch_clips, batch_labels in context_clip_loader:

                    batch_clips = batch_clips.to(self.device)
                    
                    sz = batch_clips.size()
                    batch_clips = batch_clips.view(-1, sz[-3], sz[-2], sz[-1]) if batch_clips.dim() >= 5 else batch_clips
                    batch_features = self.model_image(batch_clips)
                    
                    features.append(batch_features)

                features = torch.cat(features, dim=0)
                features = self.pooler(features)

                features = features.detach()

                self.classifier.configure(features, batch_labels, None, object_list=object_list)
                gc.collect()




            """

            inner_loop_optimizer = init_optimizer(self, lr, optimizer_type, extractor_scale_factor)
                
            for _ in range(self.args.num_grad_steps):
                for batch_context_clips, batch_context_labels in context_clip_loader:
                    
                    batch_features = self.model_image(batch_context_clips)
                    batch_features = self.pooler(batch_features)
                    batch_logits = self.classifier.predict(batch_features, None)

                    batch_loss = cross_entropy(batch_logits, batch_context_labels)
                    batch_loss.backward()
                    
                    # inner pass through updating
                    #inner_loop_optimizer.step()
                    #inner_loop_optimizer.zero_grad()

                #post pass through updating
                inner_loop_optimizer.step()
                inner_loop_optimizer.zero_grad()

            """

            self.eval_test(target_frames_by_video, target_paths_by_video, target_labels_by_video)
            """
            with torch.no_grad():
                #for video_frames, video_paths, video_label in zip(cached_target_frames_by_video, cached_target_paths_by_video, cached_target_labels_by_video):
                for video_frames, video_paths, video_label in zip(target_frames_by_video, target_paths_by_video, target_labels_by_video):
                    #print(video_frames.shape)
                    video_clips = attach_frame_history(video_frames, self.args.clip_length)
                    #print(self.args.clip_length)
                    print(video_clips.shape)
                    clip_loader = get_clip_loader(video_clips, self.batch_size)
                    #clip_loader = get_clip_loader(video_frames, self.batch_size)

                    features = []

                    for batch_clips in clip_loader:

                        sz = batch_clips.size()
                        
                        #batch_clips = batch_clips.to(torch.device("cuda:0"))
                        batch_clips = batch_clips.to(self.device)
                        batch_clips = batch_clips.view(-1, sz[-3], sz[-2], sz[-1]) if batch_clips.dim() >= 5 else batch_clips
                        batch_features = self.model_image(batch_clips)

                        #gc.collect()
                        
                        #features.append(batch_features)

                    del video_clips
                    del video_frames
                    del clip_loader
                    #gc.collect()
                    #features = torch.cat(features, dim=0)
                    #features = self.pooler(features)
                    #video_logits = self.classifier.predict(features)
            """

            """
                    self.test_evaluator.append_video(video_logits, video_label, video_paths, object_list)


                if (step + 1) & self.args.test_tasks_per_user == 0:
                    _, current_user_stats = self.test_evaluator.get_mean_stats(current_user=True)
                    print_and_log(self.logfile, f'{self.args.test_set} user {task_dict["user_id"]} ({self.test_evaluator.current_user+1}/{self.test_queue.num_users}) stats: {stats_to_str(current_user_stats)}') 

                    if (step+1) <  num_test_tasks:
                        self.test_evaluator.next_user()
            

        stats_per_user, stats_per_video = self.test_evaluator.get_mean_stats()
        stats_per_user_str, stats_per_video_str = stats_to_str(stats_per_user), stats_to_str(stats_per_video)
        print_and_log(self.logfile, f'{self.args.test_set} [{path}]\n per-user stats: {stats_per_user_str}\n per-video stats: {stats_per_video_str}\n model stats: {0.0}\n')

        self.test_evaluator.save()
        self.test_evaluator.reset()
        
        """
        


                   

        


    
    def save_checkpoint(self, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_stats': self.validation_evaluator.get_current_best_stats()
        }, os.path.join(self.checkpoint_dir, 'checkpoint.pt'))

    def load_checkpoint(self):
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'checkpoint.pt'))
        self.start_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.validation_evaluator.replace(checkpoint['best_stats'])

if __name__ == "__main__":
    main()
