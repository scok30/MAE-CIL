import copy
import math

import numpy as np
import torch
from timm.models.layers import trunc_normal_
from torch import nn
import torch.fft
import continual.utils as cutils
import torch.nn.functional as F

class ContinualClassifier(nn.Module):
    """Your good old classifier to do continual."""
    def __init__(self, embed_dim, nb_classes):
        super().__init__()

        self.embed_dim = embed_dim
        self.nb_classes = nb_classes
        self.head = nn.Linear(embed_dim, nb_classes, bias=True)
        self.norm = nn.LayerNorm(embed_dim)

    def reset_parameters(self):
        self.head.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, x):
        x = self.norm(x)
        return self.head(x)

    def add_new_outputs(self, n):
        head = nn.Linear(self.embed_dim, self.nb_classes + n, bias=True)
        head.weight.data[:-n] = self.head.weight.data

        head.to(self.head.weight.device)
        self.head = head
        self.nb_classes += n

class ClassFusion(nn.Module):
    """Your good old classifier to do continual."""
    def __init__(self, embed_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim*2,embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim,embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
    def forward(self, x, shallow_feature):
        x = torch.cat([x,shallow_feature],dim=-1)
        return self.layers(x)

class DyTox(nn.Module):
    """"DyTox for the win!

    :param transformer: The base transformer.
    :param nb_classes: Thhe initial number of classes.
    :param individual_classifier: Classifier config, DyTox is in `1-1`.
    :param head_div: Whether to use the divergence head for improved diversity.
    :param head_div_mode: Use the divergence head in TRaining, FineTuning, or both.
    :param joint_tokens: Use a single TAB forward with masked attention (faster but a bit worse).
    """
    def __init__(
        self,
        transformer,
        nb_classes,
        individual_classifier='',
        head_div=False,
        head_div_mode=['tr', 'ft'],
        joint_tokens=False,
        **kwargs
    ):
        super().__init__()

        self.nb_classes = nb_classes
        self.embed_dim = transformer.embed_dim
        self.individual_classifier = individual_classifier
        self.use_head_div = head_div
        self.head_div_mode = head_div_mode
        self.head_div = None
        self.joint_tokens = joint_tokens
        self.in_finetuning = False

        self.nb_classes_per_task = [nb_classes]

        self.patch_embed = transformer.patch_embed
        self.pos_embed = transformer.pos_embed
        self.pos_drop = transformer.pos_drop
        self.sabs = transformer.blocks[:transformer.local_up_to_layer]

        self.tabs = transformer.blocks[transformer.local_up_to_layer:]
        self.mae_decoder = transformer.mae_decoder
        self.task_tokens = nn.ParameterList([transformer.cls_token])

        if self.individual_classifier != '':
            in_dim, out_dim = self._get_ind_clf_dim()
            self.head = nn.ModuleList([
                ContinualClassifier(in_dim, out_dim).cuda()
            ])
        else:
            self.head = ContinualClassifier(
                self.embed_dim * len(self.task_tokens), sum(self.nb_classes_per_task)
            ).cuda()
        self.shallow_ratio = 0.2
        self.shallow_expand = nn.Linear(int(self.shallow_ratio*self.embed_dim),self.embed_dim).cuda()
        self.class_fusion = ClassFusion(self.embed_dim).cuda()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim),requires_grad=True).cuda()
        ksz = self.patch_embed.proj.kernel_size
        self.mae_decoder_pred = nn.Linear(self.embed_dim,ksz[0]*ksz[1]*3).cuda()
        # plus
        if 'args' in kwargs:
            self.args=kwargs['args']


    def end_finetuning(self):
        """Start FT mode, usually with backbone freezed and balanced classes."""
        self.in_finetuning = False

    def begin_finetuning(self):
        """End FT mode, usually with backbone freezed and balanced classes."""
        self.in_finetuning = True

    def add_model(self, nb_new_classes):
        """Expand model as per the DyTox framework given `nb_new_classes`.

        :param nb_new_classes: Number of new classes brought by the new task.
        """
        self.nb_classes_per_task.append(nb_new_classes)

        # Class tokens ---------------------------------------------------------
        new_task_token = copy.deepcopy(self.task_tokens[-1])
        trunc_normal_(new_task_token, std=.02)
        self.task_tokens.append(new_task_token)
        # ----------------------------------------------------------------------

        # Diversity head -------------------------------------------------------
        if self.use_head_div:
            self.head_div = ContinualClassifier(
                self.embed_dim, self.nb_classes_per_task[-1] + 1
            ).cuda()
        # ----------------------------------------------------------------------

        # Classifier -----------------------------------------------------------
        if self.individual_classifier != '':
            in_dim, out_dim = self._get_ind_clf_dim()
            self.head.append(
                ContinualClassifier(in_dim, out_dim).cuda()
            )
        else:
            self.head = ContinualClassifier(
                self.embed_dim * len(self.task_tokens), sum(self.nb_classes_per_task)
            ).cuda()
        # ----------------------------------------------------------------------

    def _get_ind_clf_dim(self):
        """What are the input and output dim of classifier depending on its config.

        By default, DyTox is in 1-1.
        """
        if self.individual_classifier == '1-1':
            in_dim = self.embed_dim
            out_dim = self.nb_classes_per_task[-1]
        elif self.individual_classifier == '1-n':
            in_dim = self.embed_dim
            out_dim = sum(self.nb_classes_per_task)
        elif self.individual_classifier == 'n-n':
            in_dim = len(self.task_tokens) * self.embed_dim
            out_dim = sum(self.nb_classes_per_task)
        elif self.individual_classifier == 'n-1':
            in_dim = len(self.task_tokens) * self.embed_dim
            out_dim = self.nb_classes_per_task[-1]
        else:
            raise NotImplementedError(f'Unknown ind classifier {self.individual_classifier}')
        return in_dim, out_dim

    def freeze(self, names):
        """Choose what to freeze depending on the name of the module."""
        requires_grad = False
        cutils.freeze_parameters(self, requires_grad=not requires_grad)
        self.train()

        for name in names:
            if name == 'all':
                self.eval()
                return cutils.freeze_parameters(self)
            elif name == 'old_task_tokens':
                cutils.freeze_parameters(self.task_tokens[:-1], requires_grad=requires_grad)
            elif name == 'task_tokens':
                cutils.freeze_parameters(self.task_tokens, requires_grad=requires_grad)
            elif name == 'sab':
                self.sabs.eval()
                cutils.freeze_parameters(self.patch_embed, requires_grad=requires_grad)
                cutils.freeze_parameters(self.pos_embed, requires_grad=requires_grad)
                cutils.freeze_parameters(self.sabs, requires_grad=requires_grad)
            elif name == 'tab':
                self.tabs.eval()
                cutils.freeze_parameters(self.tabs, requires_grad=requires_grad)
            elif name == 'old_heads':
                self.head[:-1].eval()
                cutils.freeze_parameters(self.head[:-1], requires_grad=requires_grad)
            elif name == 'heads':
                self.head.eval()
                cutils.freeze_parameters(self.head, requires_grad=requires_grad)
            elif name == 'head_div':
                self.head_div.eval()
                cutils.freeze_parameters(self.head_div, requires_grad=requires_grad)
            else:
                raise NotImplementedError(f'Unknown name={name}.')

    def param_groups(self):
        return {
            'all': self.parameters(),
            'old_task_tokens': self.task_tokens[:-1],
            'task_tokens': self.task_tokens.parameters(),
            'new_task_tokens': [self.task_tokens[-1]],
            'sa': self.sabs.parameters(),
            'patch': self.patch_embed.parameters(),
            'pos': [self.pos_embed],
            'ca': self.tabs.parameters(),
            'old_heads': self.head[:-self.nb_classes_per_task[-1]].parameters() \
                              if self.individual_classifier else \
                              self.head.parameters(),
            'new_head': self.head[-1].parameters() if self.individual_classifier else self.head.parameters(),
            'head': self.head.parameters(),
            'head_div': self.head_div.parameters() if self.head_div is not None else None

        }

    def reset_classifier(self):
        if isinstance(self.head, nn.ModuleList):
            for head in self.head:
                head.reset_parameters()
        else:
            self.head.reset_parameters()

    def hook_before_update(self):
        pass

    def hook_after_update(self):
        pass

    def hook_after_epoch(self):
        pass

    def epoch_log(self):
        """Write here whatever you want to log on the internal state of the model."""
        log = {}

        # Compute mean distance between class tokens
        mean_dist, min_dist, max_dist = [], float('inf'), 0.
        with torch.no_grad():
            for i in range(len(self.task_tokens)):
                for j in range(i + 1, len(self.task_tokens)):
                    dist = torch.norm(self.task_tokens[i] - self.task_tokens[j], p=2).item()
                    mean_dist.append(dist)

                    min_dist = min(dist, min_dist)
                    max_dist = max(dist, max_dist)

        if len(mean_dist) > 0:
            mean_dist = sum(mean_dist) / len(mean_dist)
        else:
            mean_dist = 0.
            min_dist = 0.

        assert min_dist <= mean_dist <= max_dist, (min_dist, mean_dist, max_dist)
        log['token_mean_dist'] = round(mean_dist, 5)
        log['token_min_dist'] = round(min_dist, 5)
        log['token_max_dist'] = round(max_dist, 5)
        return log

    def get_internal_losses(self, clf_loss):
        """If you want to compute some internal loss, like a EWC loss for example.

        :param clf_loss: The main classification loss (if you wanted to use its gradient for example).
        :return: a dictionnary of losses, all values will be summed in the final loss.
        """
        int_losses = {}
        return int_losses

    def forward_features_vit(self, x):
        # Shared part, this is the ENCODER
        B = x.shape[0]

        x = self.patch_embed(x)
        # print(x.shape,self.pos_embed.shape)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        # print(self.task_tokens[0].shape) (1,1,384)
        for blk in self.sabs:
            x = blk(x)
            # print(x.shape) (128,64,384)

        # Specific part, this is what we called the "task specific DECODER"
        if self.joint_tokens:
            return self.forward_features_jointtokens(x)

        tokens = []
        attentions = []
        mask_heads = None
        for task_token in self.task_tokens:
            task_token = task_token.expand(B, -1, -1)

            for blk in self.tabs:
                task_token = blk(torch.cat((task_token, x), dim=1))

            tokens.append(task_token[:, 0])

        self._class_tokens = tokens
        return tokens, tokens[-1], attentions

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        if not hasattr(self,'noise'):
            self.noise = noise[0]
            for i in range(len(noise)):
                noise[i] = self.noise
        else:
            for i in range(len(noise)):
                noise[i] = self.noise
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def mae_inference(self, x):
        original_x = x.copy()
        x=x/255.
        imagenet_mean=np.array([0.485,0.456,0.406])
        imagenet_std=np.array([0.229,0.224,0.225])
        x=(x-imagenet_mean)/imagenet_std
        x=np.expand_dims(x,0)
        x=torch.tensor(x).permute(0,3,1,2).cuda().float()
        # Shared part, this is the ENCODER
        B = x.shape[0]

        x = self.patch_embed(x)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        # apply masking
        masked_x, mask, ids_restore = self.random_masking(x, mask_ratio=0.75)

        def run_encoder(x):
            s_e, s_a, s_v = [], [], []
            shallow_feature = None
            for i, blk in enumerate(self.sabs):
                x, attn, v = blk(x)
                s_e.append(x)
                s_a.append(attn)
                s_v.append(v)
                if i == 1:
                    shallow_feature = x
            return s_e, s_a, s_v, shallow_feature, x

        s_e, s_a, s_v, shallow_feature, x = run_encoder(x)
        _, _, _, masked_shallow_feature, masked_x = run_encoder(masked_x)
        mask_heads = None

        def task_decoder(x):
            tokens = []
            attentions = []
            for task_token in self.task_tokens:
                task_token = task_token.expand(B, -1, -1)

                for blk in self.tabs:
                    task_token, attn, v = blk(torch.cat((task_token, x), dim=1), mask_heads=mask_heads)

                attentions.append(attn)
                tokens.append(task_token[:, 0])

            return tokens, attentions

        def mae_decoder(x):
            x, attn, v = self.mae_decoder(x, mask_heads=mask_heads)
            return x

        def extract_high_frequency_info(imgs):
            imgs = imgs.to(torch.float32)
            f_imgs = torch.fft.fftn(imgs, dim=[2, 3])
            h, w = imgs.shape[2:]
            mask = torch.ones_like(f_imgs)
            mask[:, :, h // 4:h // 4 * 3, w // 4:w // 4 * 3] = 0
            return f_imgs * mask

        def convert2spatial(f_imgs):
            return torch.fft.ifftn(f_imgs.to(torch.float32), dim=[2, 3]).real

        tokens, attentions = task_decoder(x)
        tokens_shallow, _ = task_decoder(shallow_feature)
        for i in range(len(tokens)):
            tokens[i] = self.class_fusion(tokens[i], tokens_shallow[i])
        # append mask token before mae decoder
        mask_tokens = self.mask_token.repeat(masked_x.shape[0], ids_restore.shape[1] - masked_x.shape[1], 1).to(
            masked_x.device)
        masked_x = torch.cat([masked_x, mask_tokens], dim=1)
        masked_x = torch.gather(masked_x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,x.shape[2]))
        masked_shallow_feature = torch.cat([masked_shallow_feature, mask_tokens], dim=1)
        masked_shallow_feature = torch.gather(masked_shallow_feature, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        masked_x = mae_decoder(masked_x)
        masked_shallow_feature = mae_decoder(masked_shallow_feature)

        def decoder_feature2restored_imgs(merged_patch):
            merged_patch = self.mae_decoder_pred(merged_patch)
            restored_imgs = self.unpatchify(merged_patch)
            return restored_imgs

        restored_imgs_main = decoder_feature2restored_imgs(masked_x)
        restored_imgs_detail = decoder_feature2restored_imgs(masked_shallow_feature)
        restored_imgs_detail = convert2spatial(extract_high_frequency_info(restored_imgs_detail))
        restored_imgs = restored_imgs_main + restored_imgs_detail
        restored_imgs = restored_imgs.permute(0,2,3,1)[0].detach().cpu().numpy()*imagenet_std+imagenet_mean
        restored_imgs = np.clip(restored_imgs,0,255).astype(np.uint8)
        mask = mask.flatten().detach().cpu().numpy().astype(np.uint8)
        patch_num = int(math.sqrt(mask.shape[-1]+1e-5))
        mask=mask.reshape((patch_num,patch_num))
        h,w=restored_imgs.shape[0]//patch_num,restored_imgs.shape[1]//patch_num
        for i in range(0,restored_imgs.shape[0],h):
            for j in range(0,restored_imgs.shape[1],w):
                if mask[i//h,j//w]==0:
                    restored_imgs[i:i+h,j:j+w]=original_x[i:i+h,j:j+w]
                else:
                    pass
                    # restored_imgs[i:i+h,j:j+w]=np.clip(original_x[i:i+h,j:j+w]*0.8+restored_imgs[i:i+h,j:j+w]*0.2,0,255).astype(np.uint8)
        # print(restored_imgs)
        return restored_imgs


    def forward_main_mae(self, x, mask_ratio=0.75):
        # Shared part, this is the ENCODER
        B = x.shape[0]

        x = self.patch_embed(x)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        # apply masking
        masked_x, mask, ids_restore = self.random_masking(x, mask_ratio=mask_ratio)

        def run_encoder(x):
            s_e, s_a, s_v = [], [], []
            shallow_feature = None
            for i, blk in enumerate(self.sabs):
                x, attn, v = blk(x)
                s_e.append(x)
                s_a.append(attn)
                s_v.append(v)
                if i == 1:
                    shallow_feature = x
            return s_e, s_a, s_v, shallow_feature, x

        _, _, _, masked_shallow_feature, masked_x = run_encoder(masked_x)


        mask_heads = None


        def mae_decoder(x):
            x, attn, v = self.mae_decoder(x, mask_heads=mask_heads)
            return x

        # append mask token before mae decoder
        mask_tokens = self.mask_token.repeat(masked_x.shape[0], ids_restore.shape[1] - masked_x.shape[1], 1).to(
            masked_x.device)
        masked_x = torch.cat([masked_x, mask_tokens], dim=1)
        masked_x = torch.gather(masked_x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        masked_x = mae_decoder(masked_x)
        merged_patch = masked_x  # bs,patch**2,dim

        merged_patch = self.mae_decoder_pred(merged_patch)
        restored_imgs = self.unpatchify(merged_patch)
        return restored_imgs.detach()

    def forward_features(self, x):
        origin_x = x.clone()
        # if self.args.model=='vit':
        #     return self.forward_features_vit(x)
        # Shared part, this is the ENCODER
        B = x.shape[0]

        x = self.patch_embed(x)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        # apply masking
        masked_x, mask, ids_restore = self.random_masking(x, mask_ratio=0.75)

        def run_encoder(x):
            s_e, s_a, s_v = [], [], []
            shallow_feature = None
            for i,blk in enumerate(self.sabs):
                x, attn, v = blk(x)
                s_e.append(x)
                s_a.append(attn)
                s_v.append(v)
                if i==1:
                    shallow_feature = self.shallow_expand(x[:,:,:int(x.shape[-1]*self.shallow_ratio)].detach())
                    # print(shallow_feature.shape)
            return s_e, s_a, s_v, shallow_feature, x

        s_e, s_a, s_v, shallow_feature, x = run_encoder(x)
        _, _, _, masked_shallow_feature, masked_x = run_encoder(masked_x)

        # Specific part, this is what we called the "task specific DECODER"
        if self.joint_tokens:
            return self.forward_features_jointtokens(x)


        mask_heads = None
        def task_decoder(x):
            tokens = []
            attentions = []
            for task_token in self.task_tokens:
                task_token = task_token.expand(B, -1, -1)

                for blk in self.tabs:
                    task_token, attn, v = blk(torch.cat((task_token, x), dim=1), mask_heads=mask_heads)

                attentions.append(attn)
                tokens.append(task_token[:, 0])

            return tokens, attentions
        def mae_decoder(x):
            x, attn, v = self.mae_decoder(x, mask_heads=mask_heads)
            return x
        def pixel_mse_loss(pred,target):
            return (torch.abs(pred-target)**2).mean()
        def pixel_l1_loss(pred,target):
            return (torch.abs(pred - target)).mean() * 0.02 # for balance with main mae loss
        def extract_high_frequency_info(imgs):
            imgs=imgs.to(torch.float32)
            f_imgs = torch.fft.fftn(imgs,dim=[2,3])
            h,w=imgs.shape[2:]
            mask=torch.ones_like(f_imgs)
            mask[:,:,int(h*0.15):int(h*0.85),int(w*0.15):int(w*0.85)]=0
            return f_imgs*mask
        def convert2spatial(f_imgs):
            return torch.fft.ifftn(f_imgs.to(torch.float32),dim=[2,3]).real
        tokens, attentions = task_decoder(x)
        tokens_shallow, _ = task_decoder(shallow_feature)
        for i in range(len(tokens)):
            tokens[i] = self.class_fusion(tokens[i], tokens_shallow[i])
        # append mask token before mae decoder
        mask_tokens = self.mask_token.repeat(masked_x.shape[0], ids_restore.shape[1] - masked_x.shape[1], 1).to(masked_x.device)
        masked_x = torch.cat([masked_x,mask_tokens],dim=1)
        masked_shallow_feature=torch.cat([masked_shallow_feature,mask_tokens],dim=1)
        masked_x = torch.gather(masked_x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        masked_shallow_feature = torch.gather(masked_shallow_feature, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        masked_x = mae_decoder(masked_x)
        masked_shallow_feature=mae_decoder(masked_shallow_feature)
        def decoder_feature2restored_imgs(merged_patch):
            merged_patch = self.mae_decoder_pred(merged_patch)
            restored_imgs = self.unpatchify(merged_patch)
            return restored_imgs

        restored_imgs_main = decoder_feature2restored_imgs(masked_x)
        restored_imgs_detail = decoder_feature2restored_imgs(masked_shallow_feature)
        restored_imgs_detail = convert2spatial(extract_high_frequency_info(restored_imgs_detail))
        restored_imgs = restored_imgs_main + restored_imgs_detail
        patch_len = origin_x.shape[-1]//int(math.sqrt(mask.shape[-1]+1e-5))
        patch_num = int(math.sqrt(mask.shape[-1]+1e-5))
        mask = mask.reshape(mask.shape[0],patch_num,-1)
        for k in range(mask.shape[0]):
            for i in range(patch_num):
                for j in range(patch_num):
                    if mask[k,i,j]==0:
                        restored_imgs[k,:,i*patch_len:(i+1)*patch_len,j*patch_len:(j+1)*patch_len]=origin_x[k,:,i*patch_len:(i+1)*patch_len,j*patch_len:(j+1)*patch_len]

        # mae loss
        mae_loss = pixel_mse_loss(restored_imgs,origin_x)
        # detail loss
        main_40_restored_imgs = self.forward_main_mae(origin_x, mask_ratio=7/16) # to make patch number divisible by 4
        main_75_restored_imgs = self.forward_main_mae(origin_x,mask_ratio=0.75)

        restored_imgs_detail = decoder_feature2restored_imgs(masked_shallow_feature)
        detail_pred=extract_high_frequency_info(restored_imgs_detail)
        detail_target=extract_high_frequency_info(main_40_restored_imgs-main_75_restored_imgs)

        detail_loss = pixel_l1_loss(detail_pred,detail_target)
        detail_loss = detail_loss.abs()

        self._class_tokens = tokens

        return tokens, tokens[-1], shallow_feature, mae_loss+detail_loss

    def forward_features_jointtokens(self, x):
        """Method to do a single TAB forward with all task tokens.

        A masking is used to avoid interaction between tasks. In theory it should
        give the same results as multiple TAB forward, but in practice it's a little
        bit worse, not sure why. So if you have an idea, please tell me!
        """
        B = len(x)

        task_tokens = torch.cat(
            [task_token.expand(B, 1, -1) for task_token in self.task_tokens],
            dim=1
        )

        for blk in self.tabs:
            task_tokens, _, _ = blk(
                torch.cat((task_tokens, x), dim=1),
                task_index=len(self.task_tokens),
                attn_mask=True
            )

        if self.individual_classifier in ('1-1', '1-n'):
            return task_tokens.permute(1, 0, 2), task_tokens[:, -1], None
        return task_tokens.view(B, -1), task_tokens[:, -1], None

    def forward_classifier(self, tokens, last_token):
        """Once all task embeddings e_1, ..., e_t are extracted, classify.

        Classifier has different mode based on a pattern x-y:
        - x means the number of task embeddings in input
        - y means the number of task to predict

        So:
        - n-n: predicts all task given all embeddings
        But:
        - 1-1: predict 1 task given 1 embedding, which is the 'independent classifier' used in the paper.

        :param tokens: A list of all task tokens embeddings.
        :param last_token: The ultimate task token embedding from the latest task.
        """
        logits_div = None
        if self.individual_classifier != '': # 1-1
            logits = []

            for i, head in enumerate(self.head):
                if self.individual_classifier in ('1-n', '1-1'):
                    logits.append(head(tokens[i]))
                else:  # n-1, n-n
                    logits.append(head(torch.cat(tokens[:i+1], dim=1)))

            if self.individual_classifier in ('1-1', 'n-1'):
                logits = torch.cat(logits, dim=1)
            else:  # 1-n, n-n
                final_logits = torch.zeros_like(logits[-1])
                for i in range(len(logits)):
                    final_logits[:, :logits[i].shape[1]] += logits[i]

                for i, c in enumerate(self.nb_classes_per_task):
                    final_logits[:, :c] /= len(self.nb_classes_per_task) - i

                logits = final_logits
        elif isinstance(tokens, torch.Tensor):
            logits = self.head(tokens)
        else:
            logits = self.head(torch.cat(tokens, dim=1))

        if self.head_div is not None and eval_training_finetuning(self.head_div_mode, self.in_finetuning):
            logits_div = self.head_div(last_token)  # only last token

        return {
            'logits': logits,
            'div': logits_div,
            'tokens': tokens
        }

    def forward(self, x):
        tokens, last_token, shallow_feature, mae_loss = self.forward_features(x)
        return self.forward_classifier(tokens, last_token), mae_loss

def eval_training_finetuning(mode, in_ft):
    if 'tr' in mode and 'ft' in mode:
        return True
    if 'tr' in mode and not in_ft:
        return True
    if 'ft' in mode and in_ft:
        return True
    return False
