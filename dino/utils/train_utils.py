import sys
import math
import tqdm
import torch

from pathlib import Path

import dino.distributed as distributed

from dino.log import MetricLogger
from dino.utils.utils import cancel_gradients_last_layer, compute_gradient_stats


def train_one_epoch(
    student,
    teacher,
    teacher_without_ddp,
    dino_loss,
    data_loader,
    optimizer,
    lr_schedule,
    wd_schedule,
    momentum_schedule,
    epoch,
    nepochs,
    fp16_scaler,
    clip_grad,
    freeze_last_layer_iter,
    save_every,
    checkpoint_dir,
    gpu_id,
    domain_classifier=None,
    domain_loss_fn=None,
    grl=None,
    dc_clip_grad: float = 0,
):
    metric_logger = MetricLogger(delimiter="  ", )
    if distributed.is_enabled():
        data_loader.sampler.set_epoch(epoch)

    with tqdm.tqdm(
        data_loader,
        desc=(f"Epoch [{epoch+1}/{nepochs}]"),
        unit=" it",
        ncols=80,
        unit_scale=1,
        total=len(data_loader),
        leave=False,
        file=sys.stdout,
        disable=not (gpu_id in [-1, 0]),
    ) as t:
        for it, (images, domain_labels) in enumerate(t):
            iteration = len(data_loader) * epoch + it # global training iteration

            # update weight decay and learning rate according to their schedule
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[iteration]
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = wd_schedule[iteration]

            # move images to gpu
            if gpu_id == -1:
                images = [im.cuda(non_blocking=True) for im in images]
                device = torch.device("cuda")
            else:
                device = torch.device(f"cuda:{gpu_id}")
                images = [im.to(device, non_blocking=True) for im in images]

            # move domain labels to gpu if doing adversarial training
            if domain_classifier is not None:
                domain_labels = domain_labels.to(device, non_blocking=True)

            # teacher and student forward passes + compute dino loss
            with torch.cuda.amp.autocast(fp16_scaler is not None):
                teacher_output = teacher(
                    images[:2]
                )  # only the 2 global views pass through the teacher

                # forward pass: extract features if doing adversarial training
                if domain_classifier is not None:
                    student_output, student_features = student(
                        images, return_features=True
                    )
                else:
                    student_output = student(images)

                loss_dino = dino_loss(student_output, teacher_output, iteration)

                # domain adversarial loss
                loss_domain = torch.tensor(0.0, device=device)
                lambda_adv = 0.0
                if domain_classifier is not None:
                    batch_size = images[0].shape[0]
                    # extract features for 2 global crops: [2*B, embed_dim]
                    global_features = student_features[: 2 * batch_size]
                    # mean-pool the 2 global crops per sample: [B, embed_dim]
                    pooled_features = global_features.view(2, batch_size, -1).mean(dim=0)
                    # apply gradient reversal and classify
                    lambda_adv = domain_loss_fn.get_lambda(iteration)
                    reversed_features = grl(pooled_features, lambda_adv)
                    domain_logits = domain_classifier(reversed_features)
                    loss_domain, _ = domain_loss_fn(
                        domain_logits, domain_labels, iteration
                    )

                loss = loss_dino + loss_domain

            if not math.isfinite(loss.item()):
                tqdm.tqdm.write(
                    "Loss is {}, stopping training".format(loss.item()), force=True
                )
                sys.exit(1)

            # student update
            optimizer.zero_grad()
            grad_stats = {}
            if fp16_scaler is None:
                loss.backward()
                # compute gradient stats and optionally clip
                grad_stats = compute_gradient_stats(
                    student, clip=clip_grad if clip_grad else None, model_name="student"
                )
                if domain_classifier is not None:
                    dc_stats = compute_gradient_stats(
                        domain_classifier,
                        clip=dc_clip_grad if dc_clip_grad else None,
                        model_name="domain_classifier",
                    )
                    grad_stats.update(dc_stats)
                cancel_gradients_last_layer(iteration, student, freeze_last_layer_iter)
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                # compute gradient stats and optionally clip
                grad_stats = compute_gradient_stats(
                    student, clip=clip_grad if clip_grad else None, model_name="student"
                )
                if domain_classifier is not None:
                    dc_stats = compute_gradient_stats(
                        domain_classifier,
                        clip=dc_clip_grad if dc_clip_grad else None,
                        model_name="domain_classifier",
                    )
                    grad_stats.update(dc_stats)
                cancel_gradients_last_layer(iteration, student, freeze_last_layer_iter)
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            # EMA update for the teacher
            with torch.no_grad():
                m = momentum_schedule[iteration]  # momentum parameter
                if torch.cuda.device_count() > 1:
                    student_params = student.module.parameters()
                else:
                    student_params = student.parameters()
                for param_q, param_k in zip(
                    student_params, teacher_without_ddp.parameters()
                ):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            # logging
            torch.cuda.synchronize()
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
            if domain_classifier is not None:
                metric_logger.update(loss_dino=loss_dino.item())
                metric_logger.update(loss_domain=loss_domain.item())
                metric_logger.update(lambda_adv=lambda_adv)

            # save checkpoint
            if distributed.is_main_process() and (iteration % save_every == 0):
                checkpoint = {
                    "iteration": iteration,
                    "student": student.state_dict(),
                    "teacher": teacher.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "dino_loss": dino_loss.state_dict(),
                }
                if fp16_scaler is not None:
                    checkpoint["fp16_scaler"] = fp16_scaler.state_dict()
                if domain_classifier is not None:
                    checkpoint["domain_classifier"] = domain_classifier.state_dict()
                save_path = Path(checkpoint_dir, f"iter_{iteration:06}.pt")
                torch.save(checkpoint, save_path)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes(gpu_id)
    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return train_stats
