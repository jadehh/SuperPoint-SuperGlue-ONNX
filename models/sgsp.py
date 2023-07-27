#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : sgsp.py
# @Author   : jade
# @Date     : 2023/7/26 15:03
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
import torch
from models.superglue import SuperGlue
from models.superpoint import SuperPoint
import torch
import torch.nn as nn
import torch.nn.functional as F


class SPSG(nn.Module):  #
    def __init__(self,sp_model_path,sg_model_path,width=320,height=320,max_keypoint=100):
        super(SPSG, self).__init__()
        self.sp_model = SuperPoint(sp_model_path,max_keypoint)
        self.sg_model = SuperGlue(sg_model_path,width,height)

    def forward(self, x1, x2):
        keypoints1, scores1, descriptors1 = self.sp_model(x1)
        keypoints2, scores2, descriptors2 = self.sp_model(x2)
        # print(scores1.shape,keypoints1.shape,descriptors1.shape)
        # example=(descriptors1.unsqueeze(0),descriptors2.unsqueeze(0),keypoints1.unsqueeze(0),keypoints2.unsqueeze(0),scores1.unsqueeze(0),scores2.unsqueeze(0))
        example = (keypoints1, scores1,descriptors1, keypoints2, scores2, descriptors2)
        indices0, indices1, mscores0, mscores1 = self.sg_model(*example)
        # return indices0,  indices1,  mscores0,  mscores1
        matches = indices0[0]

        valid = torch.nonzero(matches > -1).squeeze().detach()
        mkpts0 = keypoints1[0].index_select(0, valid);
        mkpts1 = keypoints2[0].index_select(0, matches.index_select(0, valid));
        confidence = mscores0[0].index_select(0, valid);
        return mkpts0, mkpts1, confidence
