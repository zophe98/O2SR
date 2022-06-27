import torch
from torch import nn
from .model_utils import init_modules
from .encoder import QuestionEncoder
from .decoder import AnswerDecoder
from .pose_embedding import positionalencoding1d, torch_extract_box_embedding
from .gcn import ContextSTGCN as STGCN
from .gcn import FrameGCN
from .linear_weightdrop import WeightDropLinear
from .fusion_modules import QuestionGuildeSpatialFusion, QuestionGuildeTemporalFusion

class _BatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features):
        super(_BatchNorm1d, self).__init__(num_features)
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_size = input.size()
        input = input.view(-1, input_size[-1]) # (N,d)
        output = super(_BatchNorm1d, self).forward(input)
        output = output.view(*input_size)
        return output

def transfer_bbox(bbox, mode = 'xyxy'):
    # From xyxy to xywh
    if mode == 'xywh':
        return bbox
    bbox[:,:,:,2] = bbox[:,:,:,2] - bbox[:,:,:,0]
    bbox[:,:,:,3] = bbox[:,:,:,3] - bbox[:,:,:,1]

    return bbox

class AMFusion(nn.Module):
    def __init__(self, module_dim):
        super(AMFusion, self).__init__()
    def forward(self, appr_feature, motion_feature, glob_question_feature):
        out = torch.cat((appr_feature, motion_feature), dim=-1)
        return out

class stgcn(nn.Module):
    def __init__(self,
                 vocab_size=-1,
                 num_classes=-1,
                 word_embedding_dim=300,
                 vision_dim=2048,
                 module_dim = 512,
                 dropout=0.1,
                 task='action',
                 use_box=True,
                 use_temporal_code = True,
                 pose_dim=4,
                 pose_mode='xywh',
                 word_embedding = None,
                 num_object=5,
                 max_video_len=35,
                 spatial_pooling='softmax',  # gumbel_softmax
                 temperature=1,
                 sin_box_embedding=False,
                 **kwargs):

        super(stgcn, self).__init__()

        spatial_code_dim = 128
        temporal_code_dim = 128

        self.appr_frame_level = True
        self.motion_frame_level = True

        self.use_box = use_box
        self.use_temporal_code = use_temporal_code
        self.num_classes=num_classes
        self.task = task

        self.pose_mode = pose_mode
        self.num_object = num_object
        self.max_video_len = max_video_len
        self.sin_box_embedding = sin_box_embedding

        self.local_glob_shared = True

        if word_embedding is not None:
            self.word_embedding = nn.Embedding.from_pretrained(word_embedding,freeze=False)
        else:
            self.word_embedding = nn.Embedding(vocab_size, word_embedding_dim)
            nn.init.uniform_(self.word_embedding.weight, -1.0, 1.0)
        self.question_encoder = QuestionEncoder(vocab_size,word_embedding_dim,module_dim,embedding=self.word_embedding)

        if self.use_box:
            if not sin_box_embedding:
                    self.box_encoder = nn.Sequential(
                        nn.Linear(pose_dim, 64),
                        _BatchNorm1d(64),
                        nn.ReLU(),
                        nn.Linear(64, spatial_code_dim),
                        _BatchNorm1d(spatial_code_dim),
                        nn.ReLU(),
                    )

        # ==== Appearance branch ====
        # ---- Generate object spatio-temporal graph nodes ----
        self.appr_feature_fc_local = WeightDropLinear(
            vision_dim,
            module_dim,
            weight_dropout=0.3,
            bias=False
        )
        if not self.local_glob_shared:
            self.appr_feature_fc_glob = WeightDropLinear(
                vision_dim,
                module_dim,
                weight_dropout=0.3,
                bias=False
            )
        appr_node_dim = module_dim
        if self.use_box:
            appr_node_dim += spatial_code_dim
        if self.use_temporal_code:
            appr_node_dim += temporal_code_dim

        self.appr_obj_feature_fc = nn.Linear(appr_node_dim, module_dim)
        self.appr_frame_feature_fc = nn.Linear(module_dim + temporal_code_dim, module_dim)

        # ==== Motion branch ====
        # ---- Generate object spatio-temporal graph nodes ----
        self.motion_feature_fc_local = WeightDropLinear(
            vision_dim,
            module_dim,
            weight_dropout=0.3,
            bias=False
        )
        if not self.local_glob_shared:
            self.motion_feature_fc_glob = WeightDropLinear(
                vision_dim,
                module_dim,
                weight_dropout=0.3,
                bias=False
            )

        motion_node_dim = module_dim
        if self.use_box:
            motion_node_dim += spatial_code_dim
        if self.use_temporal_code:
            motion_node_dim += temporal_code_dim
        self.motion_obj_feature_fc = nn.Linear(motion_node_dim, module_dim)
        self.motion_frame_feature_fc = nn.Linear(module_dim + temporal_code_dim, module_dim)

        # ====  Generate scene node ====
        self.appr_visual_fc = nn.Sequential(
            nn.Linear(module_dim * 2, module_dim),
        )
        self.motion_visual_fc = nn.Sequential(
            nn.Linear(module_dim * 2, module_dim),
        )

        #  (1, TN*TN, TN*TN)
        self.spatial_mask, self.temporal_mask = \
            self.make_sparse_temporal_graph(self.max_video_len, self.num_object, temporal_window=3, upper=False)

        # ==== Appr branch object-level relational reasoning ====
        self.appr_st_gcn = STGCN(
                module_dim,
                num_layer=1,
                temperature=temperature,
                graph_mask=self.spatial_mask,
                fixed_graph=False,
                num_head=1,
                dropout=dropout,
                q2v_flag=True)

        self.appr_spatial_pooling = QuestionGuildeSpatialFusion(module_dim, num_head=1,attn=spatial_pooling)
        self.appr_temporal_pooling = QuestionGuildeTemporalFusion(module_dim)

        # ==== Motion branch object-level relational reasoning ====
        self.motion_st_gcn = STGCN(
                module_dim,
                num_layer=1,
                temperature=temperature,
                graph_mask=self.spatial_mask,
                fixed_graph=False,
                num_head=1,
                dropout=dropout,
                q2v_flag=True)

        self.motion_spatial_pooling = QuestionGuildeSpatialFusion(module_dim, num_head=1,
                                                                  attn=spatial_pooling)
        self.motion_temporal_pooling = QuestionGuildeTemporalFusion(module_dim)

        # ==== scene-level ralational reasoning ====
        self.appr_frame_gcn = FrameGCN(module_dim,dropout=dropout,num_head=1, question2visual_flag=True, norm=True)
        self.motion_frame_gcn = FrameGCN(module_dim,dropout=dropout,num_head=1, question2visual_flag=True, norm=True)

        self.qv_fusion = AMFusion(module_dim)

        self.answer_output_unit = AnswerDecoder(task,module_dim,num_classes, input_dim= module_dim * 2)
        init_modules(self.modules(), w_init='xavier_uniform')


        temporal_code = positionalencoding1d(temporal_code_dim, self.max_video_len)
        # (1, max_len,d)
        temporal_code = (temporal_code).unsqueeze(0).unsqueeze(2)
        # for move into cuda later
        self.temporal_code = nn.Parameter(temporal_code, requires_grad=True) # dynamically-learned
        # self.register_buffer('temporal_code', temporal_code)

        glob_temporal_code = positionalencoding1d(temporal_code_dim, self.max_video_len)
        # (1, max_len,d)
        glob_temporal_code = (glob_temporal_code).unsqueeze(0)
        # for move into cuda later
        self.glob_temporal_code = nn.Parameter(glob_temporal_code, requires_grad=True)
        # self.register_buffer('glob_temporal_code', glob_temporal_code)

    # (1, T*N, T*N)
    def make_full_temporal_graph(self,temporal_dim, spatial_dim, temporal_window=3):
        # (1, N, T*N)
        base_adja = torch.zeros((1, spatial_dim, temporal_dim * spatial_dim))
        base_adja[0, :, :temporal_window * spatial_dim] = 1
        # (T, N, T*N)
        full_adj = base_adja.repeat(temporal_dim, 1, 1)
        # roll
        for i in range(temporal_dim):
            roll_shift = (i - 1) * spatial_dim
            full_adj[i] = torch.roll(full_adj[i], roll_shift, -1)
        # mask the lower left and upper right
        full_adj[(temporal_dim - 1), :, :spatial_dim] = 0
        full_adj[0, :, (temporal_dim - 1) * spatial_dim - 1:] = 0

        full_adj = full_adj.view(1, temporal_dim * spatial_dim, temporal_dim * spatial_dim)
        return full_adj

    # (1, T*N,T*N)
    def make_diagonal_spatial_graph(self,temporal_dim, spatial_dim, temporal_window=1):
        # (1, N, T*N)
        base_adja = torch.zeros((1, spatial_dim, temporal_dim * spatial_dim))
        base_adja[0, :, :temporal_window * spatial_dim] = 1
        # (T, N, T*N)
        diagonal_adj = base_adja.repeat(temporal_dim, 1, 1)
        # roll
        for i in range(temporal_dim):
            roll_shift = i * spatial_dim
            diagonal_adj[i] = torch.roll(diagonal_adj[i], roll_shift, -1)

        diagonal_adj = diagonal_adj.view(1, temporal_dim * spatial_dim, temporal_dim * spatial_dim)
        return diagonal_adj

    # (1, T*N, T*N)
    def make_sparse_temporal_graph(self, temporal_dim, spatial_dim, temporal_window=3, upper=False):
        full_graph = self.make_full_temporal_graph(temporal_dim, spatial_dim, temporal_window)

        return full_graph, full_graph


    def model_block(self,
                    linguistic, linguistic_len,linguistic_mask,
                    visual_feature,appr_glob_feature,
                    motion_feature, motion_glob_feature,
                    visual_mask, bbox = None):
        """
        Args:
            linguistic          (B,L)
            linguistic_len      (B,)
            linguistic_mask     (B,L)
            visual_feature      (B,T,N,D)
            appr_glob_feature   (B,T,D)
            motion_feature      (B,T,N,D)
            motion_glob_feature (B,T,D)
            visual_mask         (B,T,N)
            bbox    (B,T,N,4)
        Returns:
             (B,T,D)
        """
        # (B,L,d) + (B,L) -> (B,L,D) + (B,D)
        linguistic_local_embedding, linguistic_glob_embedding = self.question_encoder(linguistic, linguistic_len)

        appr_st_feature = visual_feature
        appr_st_feature, appr_cross_adj, appr_spatial_adj, appr_temporal_adj = self.appr_st_gcn(
                        linguistic_local_embedding, linguistic_glob_embedding,linguistic_mask,
                        appr_st_feature, visual_mask,None,None)

        motion_st_feature = motion_feature
        motion_st_feature, motion_cross_adj,motion_spatial_adj, motion_temporal_adj = \
                self.motion_st_gcn(linguistic_local_embedding, linguistic_glob_embedding,linguistic_mask,
                                                    motion_st_feature,visual_mask,None,None)

        appr_avg_object_feature = self.appr_spatial_pooling(linguistic_glob_embedding, appr_st_feature, visual_mask)

        motion_avg_object_feature = self.motion_spatial_pooling(linguistic_glob_embedding,motion_st_feature,visual_mask)

        # ====  Generate scene node ====
        appr_avg_object_feature = torch.cat((appr_avg_object_feature, appr_glob_feature), dim=-1)
        motion_avg_object_feature = torch.cat((motion_avg_object_feature, motion_glob_feature), dim=-1)
        appr_avg_object_feature = self.appr_visual_fc(appr_avg_object_feature)
        motion_avg_object_feature = self.motion_visual_fc(motion_avg_object_feature)

        # ==== Scene-level realtional reasoning
        appr_avg_object_feature, appr_temporal_adj, appr_temporal_cross = self.appr_frame_gcn(
            appr_avg_object_feature,visual_mask.sum(-1),
            linguistic_local_embedding,linguistic_mask,linguistic_glob_embedding)

        motion_avg_object_feature, motion_temporal_adj, motion_temporal_cross = self.motion_frame_gcn(
            motion_avg_object_feature,visual_mask.sum(-1),
            linguistic_local_embedding,linguistic_mask,linguistic_glob_embedding)

        # ====  Multimodal fusion  ====
        appr_answer_feature, appr_temporal_attn = self.appr_temporal_pooling(
                    linguistic_glob_embedding,appr_avg_object_feature,visual_mask.sum(-1),True)

        motion_answer_feature, motion_temporal_attn = self.motion_temporal_pooling(
            linguistic_glob_embedding,motion_avg_object_feature, visual_mask.sum(-1), True)

        answer_feature = self.qv_fusion(appr_answer_feature, motion_answer_feature, linguistic_glob_embedding)


        # 将跨模态和注意力权重展示出来
        appr_cross_adj = appr_cross_adj.squeeze(1) # (B,1,TN,L) -> (B,TN,L)
        motion_cross_adj = motion_cross_adj.squeeze(1)

        appr_temporal_cross = appr_temporal_cross.squeeze(1)        # (bs, 1, T, L) -> (bs, T, L)
        motion_temporal_cross = motion_temporal_cross.squeeze(1)

        return answer_feature, linguistic_glob_embedding, \
                appr_cross_adj, motion_cross_adj, \
                appr_temporal_cross, motion_temporal_cross, \
                appr_temporal_attn, motion_temporal_attn

    def forward_trans_or_action(self,
                    questions, questions_len,questions_mask,
                    visual_features, appr_glob_feature,
                    motion_feature, motion_glob_feature,
                    visual_masks, bbox = None, answers = None):

        out = {}

        # (B,5,L)   -> (5,B,L)
        questions = questions.transpose(0, 1)
        # (B,5)     -> (5,B)
        questions_len = questions_len.transpose(0, 1)
        # (B,5,L)   -> (5,B,L)
        questions_mask = questions_mask.transpose(0, 1)

        av_features = []
        q_features = []

        appr_obj_crosses, motion_obj_crosses = [], []
        appr_frame_crosses, motion_frame_crosses = [], []
        appr_attens, motion_attens = [], []

        for idx, candidate in enumerate(questions):
            av_feature,q_feature, \
            appr_obj_cross,motion_obj_cross,\
            appr_frame_cross,motion_frame_cross,\
            appr_attn, motion_attn =  self.model_block(
                                            candidate,questions_len[idx],questions_mask[idx],
                                            visual_features,appr_glob_feature,
                                            motion_feature,motion_glob_feature,
                                            visual_masks,bbox)
            av_features.append(av_feature)
            q_features.append(q_feature)

            appr_obj_crosses.append(appr_obj_cross)
            motion_obj_crosses.append(motion_obj_cross)

            appr_frame_crosses.append(appr_frame_cross)
            motion_frame_crosses.append(motion_frame_cross)

            appr_attens.append(appr_attn)
            motion_attens.append(motion_attn)

        # (5,B,D) -> (B,5,D)
        av_features = torch.stack(av_features,0).transpose(0,1)

        # (5,B,D) -> (B,5,D)
        q_features = torch.stack(q_features,0).transpose(0,1)

        # stack the adjs
        appr_obj_crosses = torch.stack(appr_obj_crosses, 0) # (5, B, TN, L)
        motion_obj_crosses = torch.stack(motion_obj_crosses, 0)  # (5, B, TN, L)

        appr_frame_crosses = torch.stack(appr_frame_crosses, 0)  # (5, B, T, L)
        motion_frame_crosses = torch.stack(motion_frame_crosses, 0)  # (5, B, T, L)

        appr_attens = torch.stack(appr_attens, 0)  # (5, B, T, 1)
        motion_attens = torch.stack(motion_attens, 0)  # (5, B, T, 1)

        # -- sample ----
        sample_index = answers.view(1,appr_obj_crosses.size(1),1,1).repeat(1,1,appr_obj_crosses.size(2),appr_obj_crosses.size(3))
        appr_obj_crosses = appr_obj_crosses.gather(0, sample_index).squeeze(0)  # (B,TN,L)
        motion_obj_crosses = motion_obj_crosses.gather(0, sample_index).squeeze(0)  # (B,TN,L)

        sample_index = answers.view(1, appr_frame_crosses.size(1), 1, 1).repeat(1, 1, appr_frame_crosses.size(2),appr_frame_crosses.size(3))
        appr_frame_crosses = appr_frame_crosses.gather(0, sample_index).squeeze(0)  # (B,T,L)
        motion_frame_crosses = motion_frame_crosses.gather(0, sample_index).squeeze(0)  # (B,T,L)

        sample_index = answers.view(1, appr_attens.size(1), 1, 1).repeat(1, 1, appr_attens.size(2),1)
        appr_attens = appr_attens.gather(0, sample_index).squeeze(0)  # (B,T,1)
        motion_attens = motion_attens.gather(0, sample_index).squeeze(0)  # (B,T,1)

        # (B,5,1)
        out['answer'] = self.answer_output_unit(q_features,av_features)


        out['object_cross'] = (appr_obj_crosses, motion_obj_crosses)    # (B, TN, L)
        out['frame_cross'] = (appr_frame_crosses, motion_frame_crosses) # (B,T,L)
        out['temporal_atten'] = (appr_attens, motion_attens)            # (B,T,1)

        return out

    def forward_frameqa(self, questions,questions_len,questions_mask,
                visual_features,appr_glob_feature,
                motion_feature, motion_glob_feature,
                visual_masks, bbox = None):

        out = {}
        qv_feature, q_feature, \
        appr_obj_cross, motion_obj_cross, \
        appr_frame_cross, motion_frame_cross, \
        appr_attn, motion_attn = self.model_block(questions, questions_len, questions_mask,
                                                                  visual_features, appr_glob_feature,
                                                                  motion_feature, motion_glob_feature,
                                                                  visual_masks, bbox)
        out['answer'] = self.answer_output_unit(q_feature,qv_feature)

        return out

    def forward(self, appr_glob_feature, appr_object_feature,
                motion_glob_feature, motion_object_feature,
                bbox, obj_mask,
                questions, questions_len, questions_mask, answers, *kargs):
        """
                                        多项选择          开放问答
        padded_glob_features        (B,T,d)         (B,T,d)
        padded_obj_features         (B,T,N,d)       (B,T,N,d)
        padded_bbox                 (B,T,N,4/6)     (B,T,N,4/6)
        obj_mask                    (B,T,N)         (B,T,N)
        all_question                (B,5,L)         (B,L)
        all_question_len            (B,5)           (B,)
        question_mask               (B,5,L)         (B,L)
        all_answers                 (B,)            (B,)
        """

        appr_object_feature = self.appr_feature_fc_local(appr_object_feature)
        if not self.local_glob_shared:
            appr_glob_feature = self.appr_feature_fc_glob(appr_glob_feature)
        else:
            appr_glob_feature = self.appr_feature_fc_local(appr_glob_feature)

        motion_object_feature = self.motion_feature_fc_local(motion_object_feature)
        if not self.local_glob_shared:
            motion_glob_feature = self.motion_feature_fc_glob(motion_glob_feature)
        else:
            motion_glob_feature = self.motion_feature_fc_local(motion_glob_feature)

        if self.use_box:
            bbox_xywh = transfer_bbox(bbox, self.pose_mode)
            if not self.sin_box_embedding:
                box_embedding = self.box_encoder(bbox_xywh)
            else:
                box_embedding = torch_extract_box_embedding(bbox_xywh, feat_dim=64, device=bbox_xywh.device)

            box_embedding = box_embedding.masked_fill(obj_mask.unsqueeze(-1) == 0, 0.0)

            appr_object_feature = torch.cat((appr_object_feature, box_embedding), dim=-1)
            motion_object_feature = torch.cat((motion_object_feature, box_embedding), dim=-1)

        if self.use_temporal_code:
            appr_temporal_feature = self.temporal_code[:, :appr_object_feature.size(1), :, :]
            temporal_feature = appr_temporal_feature.repeat(appr_object_feature.size(0), 1, appr_object_feature.size(2),1)

            appr_object_feature = torch.cat((appr_object_feature, temporal_feature), dim=-1)
            motion_object_feature = torch.cat((motion_object_feature, temporal_feature), dim=-1)

            appr_frame_code_feature = self.glob_temporal_code[:, :appr_glob_feature.size(1), :].repeat(appr_glob_feature.size(0), 1, 1)
            motion_frame_code_feature = self.glob_temporal_code[:, :motion_glob_feature.size(1),:].repeat(motion_glob_feature.size(0), 1, 1)

            appr_glob_feature = torch.cat((appr_glob_feature, appr_frame_code_feature), dim=-1)
            motion_glob_feature =  torch.cat((motion_glob_feature, motion_frame_code_feature), dim=-1)

            appr_glob_feature = self.appr_frame_feature_fc(appr_glob_feature)
            motion_glob_feature = self.motion_frame_feature_fc(motion_glob_feature)


        # (B,T,N,k*D) -> (B,T,N,D)
        appr_object_feature = self.appr_obj_feature_fc(appr_object_feature)
        motion_object_feature = self.motion_obj_feature_fc(motion_object_feature)

        if self.task == 'frameqa' or self.task == 'count':
            # (B,num_class)  num_class = 1 or num_class = (answer set len)
            out = self.forward_frameqa(questions, questions_len, questions_mask,
                                       appr_object_feature, appr_glob_feature,
                                       motion_object_feature,motion_glob_feature,
                                       obj_mask, bbox)

        elif self.task == 'action' or self.task == 'transition':
            # (B,5,1)
            out = self.forward_trans_or_action(
                questions, questions_len, questions_mask,
                appr_object_feature, appr_glob_feature,
                motion_object_feature,motion_glob_feature,
                obj_mask, bbox, answers)
        else:
            raise ValueError("No support task {}".format(self.task))
        # {'caption', 'answer'}
        return out, answers
