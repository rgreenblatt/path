import os
import sys
import datetime
import math

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from criterion import PerceptualLoss
from model import Net
from apex.parallel import DistributedDataParallel, convert_syncbn_model
from apex import amp, optimizers
import git

from config import Config
from torch_utils import (LRSched, LossTracker, EMATracker,
                         recursive_param_print)
from utils import mkdirs, PrintAndLog
import neural_render_generate_data


def ceildiv(a, b):
    return -(-a // b)


def make_divisible(a, b):
    return ceildiv(a, b) * b


def main():
    cfg = Config()

    disable_all_output = cfg.local_rank != 0

    logger = None
    writer = None
    if not disable_all_output:
        output_dir = os.path.join("outputs", cfg.name)
        tensorboard_output = os.path.join(output_dir, "tensorboard")
        model_save_output = os.path.join(output_dir, "models")

        if os.path.exists(output_dir):
            print("output directory exists, returning")
            sys.exit(1)

        mkdirs(tensorboard_output)
        mkdirs(model_save_output)

        writer = SummaryWriter(log_dir=tensorboard_output)

        logger = PrintAndLog(os.path.join(output_dir, "output.log"))

        cfg.print_params()
        cfg.print_non_default()

        writer.add_text("params", cfg.as_markdown())
        writer.add_text("non default params", cfg.non_default_as_markdown())

        try:
            repo = git.Repo(search_parent_directories=True)
            commit_hash = repo.head.object.hexsha
            writer.add_text("git commit hash", commit_hash)
            print("git commit hash:", commit_hash)
        except (git.InvalidGitRepositoryError, git.NoSuchPathError) as e:
            print("failed to get git commit hash with err:")
            print(e)

        print()

    torch.backends.cudnn.benchmark = not cfg.no_cudnn_benchmark

    use_distributed = False
    if 'WORLD_SIZE' in os.environ:
        use_distributed = int(os.environ['WORLD_SIZE']) > 1

    which_gpu = 0
    world_size = 1

    assert torch.cuda.is_available()

    if use_distributed:
        which_gpu = cfg.local_rank
        torch.cuda.set_device(which_gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        world_size = torch.distributed.get_world_size()

    torch.manual_seed(cfg.seed + which_gpu)

    def reduce_tensor(tensor):
        if not use_distributed:
            return tensor

        rt = tensor.clone()
        torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
        rt /= world_size

        return rt

    # This is redundant I think
    device = torch.device("cuda:{}".format(which_gpu))

    batch_size = cfg.batch_size

    show_model_info = cfg.show_model_info and not disable_all_output

    net = Net()

    if use_distributed and not cfg.no_sync_bn:
        net = convert_syncbn_model(net)

    lr = 0.05
    optimizer = torch.optim.LBFGS(net.parameters(), lr=lr)

    net = net.to(device)

    net, optimizer = amp.initialize(net,
                                    optimizer,
                                    opt_level=cfg.opt_level,
                                    min_loss_scale=65536.0,
                                    verbosity=cfg.amp_verbosity)

    if use_distributed:
        net = DistributedDataParallel(net)

    if cfg.no_perceptual_loss:
        criterion = torch.nn.MSELoss().to(device)
    else:
        criterion = PerceptualLoss().to(device)

    if show_model_info:
        print(net)

    if show_model_info:
        print()

        _, to_print = recursive_param_print(net)
        print(to_print)

    step = 0

    if not disable_all_output:
        print()
        print("===== Training =====")
        print()

    factor = 1e-4
    world_batch_size = batch_size * world_size

    epoch_size = make_divisible(cfg.epoch_size, world_batch_size)
    validation_size = make_divisible(cfg.validation_size, world_batch_size)
    num_local_batchs_per_epoch = epoch_size // batch_size
    num_local_batchs_per_validation = validation_size // batch_size

    scaled_lr = cfg.lr_multiplier * world_batch_size * factor
    lr_schedule = LRSched(scaled_lr, cfg.epochs * epoch_size)

    train_seed_start = 0
    validation_seed_start = 2**30
    image_seed = validation_seed_start - 1

    norm_avg = EMATracker(0.9, 50.0)

    for epoch in range(cfg.epochs):
        net.train()

        i = 0

        train_loss_tracker = LossTracker(reduce_tensor)

        steps_since_display = 0
        max_train_step = epoch_size
        format_len = math.floor(math.log10(max_train_step)) + 1

        def get_or_nan(v):
            if v is None:
                return float('nan')
            else:
                return v

        def train_display():
            train_loss, nan_count = train_loss_tracker.query_reset()
            if not disable_all_output:
                print("{}, epoch {}/{}, step {}/{}, train loss {:.4e}, NaN {}".
                      format(datetime.datetime.now(), epoch, cfg.epochs - 1,
                             str((i + 1) * world_batch_size).zfill(format_len),
                             max_train_step, get_or_nan(train_loss),
                             nan_count * world_batch_size),
                      flush=True)
                if train_loss is not None:
                    writer.add_scalar("loss/train", train_loss, step)
                writer.add_scalar("lr", lr, step)
                writer.flush()

        # TODO: consider some sort of visualization tracking!

        steps_since_set_lr = 0

        def set_lr():
            nonlocal lr
            lr = lr_schedule(step)

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        set_lr()

        for i, base_seed in enumerate(
                range(which_gpu, num_local_batchs_per_epoch, world_size)):
            seed = base_seed * world_batch_size + train_seed_start

            # this wouldn't be strictly accurate if we had partial batches
            step += world_batch_size
            steps_since_display += world_batch_size
            steps_since_set_lr += world_batch_size

            (scenes, coords, values) = neural_render_generate_data.gen_data(
                batch_size, cfg.rays_per_tri, cfg.samples_per_ray, seed)

            (scenes, coords, values) = (scenes.to(device), coords.to(device),
                                        values.to(device))

            is_first = True

            class NanLoss(Exception):
                pass

            def get_loss():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()

                outputs = net(scenes, coords)
                loss = criterion(outputs, values)

                nonlocal is_first
                if is_first:
                    is_nan = train_loss_tracker.update(loss)
                    if is_nan:
                        raise NanLoss()

                if torch.is_grad_enabled():
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()

                    if is_first:
                        max_norm = norm_avg.x * 1.5
                        this_norm = nn.utils.clip_grad_norm_(
                            net.parameters(), max_norm)
                        norm_avg.update(this_norm)
                        if not disable_all_output:
                            writer.add_scalar("max_norm", max_norm, step)
                            writer.add_scalar("norm", this_norm, step)

                is_first = False

                return loss

            try:
                optimizer.step(get_loss)
            except NanLoss:
                continue

            if steps_since_set_lr >= cfg.set_lr_freq:
                set_lr()
                steps_since_set_lr = 0

            if steps_since_display >= cfg.display_freq:
                train_display()
                steps_since_display = 0

        train_seed_start += num_local_batchs_per_epoch * world_batch_size

        if steps_since_display > 0:
            train_display()

        net.eval()

        test_loss_tracker = LossTracker(reduce_tensor)

        with torch.no_grad():
            for i, base_seed in enumerate(
                    range(which_gpu, num_local_batchs_per_validation,
                          world_size)):
                seed = base_seed * world_batch_size + validation_seed_start
                (scenes, coords,
                 values) = neural_render_generate_data.gen_data(
                     batch_size, cfg.rays_per_tri, cfg.samples_per_ray, seed)

                (scenes, coords,
                 values) = (scenes.to(device), coords.to(device),
                            values.to(device))
                outputs = net(scenes, coords)
                loss = criterion(outputs, values)

                test_loss_tracker.update(loss)

            (scenes, coords, values,
             indexes) = neural_render_generate_data.gen_data_for_image(
                 cfg.image_count, cfg.image_dim, cfg.samples_per_ray,
                 image_seed)

            (scenes, coords) = (scenes.to(device), coords.to(device))

            outputs = net(scenes, coords)
            actual_img = torch.zeros(cfg.image_count, cfg.image_dim,
                                     cfg.image_dim, 3)
            output_img = torch.zeros(cfg.image_count, cfg.image_dim,
                                     cfg.image_dim, 3)
            actual_img[:, indexes[:, 1], indexes[:, 0]] = values
            output_img[:, indexes[:, 1], indexes[:, 0]] = outputs.cpu()

            def tone_map(x):
                return x / (x + 1)

            writer.add_image("images/actual",
                             make_grid(tone_map(actual_img.moveaxis(-1, 1))),
                             step)
            writer.add_image("images/output",
                             make_grid(tone_map(output_img.moveaxis(-1, 1))),
                             step)

        test_loss, count_nan = test_loss_tracker.query_reset()
        if not disable_all_output:
            print(
                "{}, epoch {}/{}, lr {:.4e}, test loss {:.4e}, NaN {}".format(
                    datetime.datetime.now(), epoch, cfg.epochs - 1, lr,
                    get_or_nan(test_loss), count_nan * world_batch_size),
                flush=True)

            if test_loss is not None:
                writer.add_scalar("loss/test", test_loss, step)

        # if not disable_all_output and (epoch + 1) % cfg.save_model_every == 0:
        #     torch.save(
        #         net, os.path.join(model_save_output, "net_{}.p".format(epoch)))

    # if not disable_all_output:
    #     torch.save(net, os.path.join(model_save_output, "net_final.p"))
    del logger


class Deinit(object):
    @staticmethod
    def __enter__():
        pass

    @staticmethod
    def __exit__(*_):
        neural_render_generate_data.deinit_renderers()


if __name__ == "__main__":
    neural_render_generate_data.deinit_renderers()
    with Deinit():
        main()
