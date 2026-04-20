import os
import torch
from tqdm import tqdm
from .utils import get_lr
import numpy as np
from .utils_bbox import decode_outputs, non_max_suppression


# 构建伪标签
def _build_pseudo_labels(outputs, input_shape, num_classes, device, conf_thres=0.6, nms_thres=0.4, min_box=1, letterbox_image=True):
    preds = decode_outputs(outputs, input_shape)
    dets = non_max_suppression(
        preds,
        num_classes,
        input_shape,
        input_shape,
        letterbox_image,
        conf_thres=conf_thres,
        nms_thres=nms_thres
    )
    labels = []
    for det in dets:
        # 空标签占位
        if det is None or len(det) == 0:
            labels.append(torch.zeros((0, 5), dtype=torch.float32, device=device))
            continue
        det = np.array(det, dtype=np.float32)
        # 左上右下坐标转中心点宽高坐标
        xyxy = det[:, :4]
        cls = det[:, 6:7]
        cxcy = (xyxy[:, 0:2] + xyxy[:, 2:4]) / 2.0
        wh = xyxy[:, 2:4] - xyxy[:, 0:2]
        # 小框过滤
        if min_box is not None:
            keep = (wh[:, 0] > min_box) & (wh[:, 1] > min_box)
            cxcy = cxcy[keep]
            wh = wh[keep]
            cls = cls[keep]
        # 空标签占位
        if cxcy.shape[0] == 0:
            labels.append(torch.zeros((0, 5), dtype=torch.float32, device=device))
            continue
        # 标签拼接
        lbl = np.concatenate([cxcy, wh, cls], axis=1).astype(np.float32)
        labels.append(torch.from_numpy(lbl).to(device))
    return labels


def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0,
                  semi_supervised=False, pseudo_conf_thres=0.6, pseudo_nms_thres=0.4, pseudo_weight=0.5, pseudo_min_box=1, pseudo_use_empty=False,
                  unsup_lang_weight=1.0):
    loss        = 0
    val_loss    = 0

    epoch_step = epoch_step // 5 

    # 进度显示
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    #

    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        if len(batch) == 7:
            images, targets, captions, multi_targets, relation, paths, is_labeled = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]
        else:
            images, targets, captions, multi_targets, relation = batch[0], batch[1], batch[2], batch[3], batch[4]
            is_labeled = [True] * images.shape[0]
        # TODO test
        # print(images.shape, targets.shape, captions.shape, multi_targets.shape, relation.shape)

        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]

        optimizer.zero_grad()
        did_step = False
        if not fp16:
            if semi_supervised:
                mask = torch.tensor(is_labeled, device=images.device, dtype=torch.bool)
                mask_list = [bool(x) for x in is_labeled]
                loss_terms = []

                # Supervised branch
                if mask.any():
                    images_l = images[mask]
                    targets_l = [t for t, m in zip(targets, mask_list) if m]
                    captions_l = [c for c, m in zip(captions, mask_list) if m]
                    multi_targets_l = [mt for mt, m in zip(multi_targets, mask_list) if m]
                    relation_l = [r for r, m in zip(relation, mask_list) if m]

                    if cuda:
                        targets_l = [ann.cuda(local_rank) for ann in targets_l]
                        captions_l = torch.tensor(np.array(captions_l)).cuda(local_rank)
                        relation_l = torch.tensor(np.array(relation_l)).cuda(local_rank)

                    outputs_l, motion_loss = model_train(images_l, captions_l, multi_targets_l, relation_l)
                    loss_sup = yolo_loss(outputs_l, targets_l) + motion_loss
                    loss_terms.append(loss_sup)

                # Unsupervised branch (pseudo labels)
                if (~mask).any():
                    images_u = images[~mask]
                    captions_u = [c for c, m in zip(captions, mask_list) if not m]
                    relation_u = [r for r, m in zip(relation, mask_list) if not m]
                    if cuda:
                        captions_u = torch.tensor(np.array(captions_u)).cuda(local_rank)
                        relation_u = torch.tensor(np.array(relation_u)).cuda(local_rank)

                    teacher = ema.ema if ema else model_train
                    teacher_was_training = teacher.training
                    teacher.eval()
                    with torch.no_grad():
                        outputs_t = teacher(images_u)
                    if teacher_was_training:
                        teacher.train()
                    input_shape = images_u.shape[-2:]
                    pseudo_labels = _build_pseudo_labels(
                        outputs_t,
                        input_shape,
                        yolo_loss.num_classes,
                        images_u.device,
                        conf_thres=pseudo_conf_thres,
                        nms_thres=pseudo_nms_thres,
                        min_box=pseudo_min_box,
                        letterbox_image=True
                    )

                    if not pseudo_use_empty:
                        keep_indices = [i for i, lbl in enumerate(pseudo_labels) if lbl.shape[0] > 0]
                    else:
                        keep_indices = list(range(len(pseudo_labels)))

                    outputs_u, motion_loss_u = model_train(images_u, captions_u, None, relation_u)
                    loss_terms.append(unsup_lang_weight * motion_loss_u)

                    if len(keep_indices) > 0:
                        outputs_u_keep = [o[keep_indices] for o in outputs_u]
                        pseudo_labels_keep = [pseudo_labels[i] for i in keep_indices]
                        loss_unsup = yolo_loss(outputs_u_keep, pseudo_labels_keep)
                        loss_terms.append(pseudo_weight * loss_unsup)

                if len(loss_terms) == 0:
                    # No valid supervised/pseudo-supervised signal in this batch.
                    loss_value = images.new_tensor(0.0)
                else:
                    loss_value = loss_terms[0]
                    for term in loss_terms[1:]:
                        loss_value = loss_value + term
            else:
                captions = torch.tensor(np.array(captions)).cuda(local_rank) if cuda else torch.tensor(np.array(captions))
                relation = torch.tensor(np.array(relation)).cuda(local_rank) if cuda else torch.tensor(np.array(relation))
                outputs, motion_loss = model_train(images, captions, multi_targets, relation)
                loss_value = yolo_loss(outputs, targets) + motion_loss

            if loss_value.requires_grad:
                loss_value.backward()
                optimizer.step()
                did_step = True
            elif local_rank == 0 and semi_supervised and iteration % max(1, epoch_step // 10) == 0:
                print("[Semi] Skip optimizer step: batch has no valid labels/pseudo-labels.")
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model_train(images) 
                loss_value = yolo_loss(outputs, targets)

            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()
            did_step = True
        if ema and did_step:
            ema.update(model_train)

        loss += loss_value.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()
        
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets = batch[0], batch[1]

        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]

            optimizer.zero_grad()
            outputs = model_train_eval(images)
            
            loss_value = yolo_loss(outputs, targets)

        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))

        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))
