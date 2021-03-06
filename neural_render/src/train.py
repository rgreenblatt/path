import os
import sys
import datetime
import math

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import git

import neural_render_generate_data_full_scene

from model import Net
from config import Config
from torch_utils import (LRSched, LossTracker, EMATracker,
                         recursive_param_print)
from utils import mkdirs, PrintAndLog
from criterion import PerceptualLoss


def ceildiv(a, b):
    return -(-a // b)


def make_divisible(a, b):
    return ceildiv(a, b) * b


def time_for_log():
    return datetime.datetime.now().replace(microsecond=0)


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
    example_device_tensor = torch.empty(0, dtype=torch.float32, device=device)

    batch_size = cfg.batch_size

    show_model_info = cfg.show_model_info and not disable_all_output

    net = Net()

    optimizer = torch.optim.LBFGS(net.parameters(),
                                  line_search_fn='strong_wolfe',
                                  history_size=10)
    net = net.to(device)

    if use_distributed:
        net = torch.nn.parallel.DistributedDataParallel(net)

    def loss_wrapper(loss):
        loss = loss.to(device)

        def get_loss(outputs, actual, count):
            # ignore first step, is just direct light
            return loss(outputs, actual[:, 1:]) / (count * cfg.rays_per_tri *
                                                   cfg.n_steps)

        return get_loss

    mse = loss_wrapper(torch.nn.MSELoss(reduction='sum'))
    perceptual = loss_wrapper(PerceptualLoss(reduction='sum'))

    if cfg.no_perceptual_loss:
        criterion = mse
    else:
        criterion = perceptual

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

    factor = 1e-3
    world_batch_size = batch_size * world_size

    epoch_size = make_divisible(cfg.epoch_size, world_batch_size)
    validation_size = make_divisible(cfg.validation_size, world_batch_size)
    num_local_batchs_per_epoch = epoch_size // batch_size
    num_local_batchs_per_validation = validation_size // batch_size

    scaled_lr = cfg.lr_multiplier * world_batch_size * factor
    # lr sched isn't very important because we use line search,
    # but it does have some effect...
    lr_schedule = LRSched(scaled_lr,
                          cfg.epochs * epoch_size,
                          pct_start=0.,
                          final_div_factor=800)

    train_seed_start = 0
    validation_seed_start = 2**30
    image_seed = validation_seed_start - 1

    norm_avg = EMATracker(0.9, 50.0)

    def save_image(name, indexes, values, counts, is_actual):
        def save_sub_image(sub_name, sub_values):
            total_img = torch.zeros(cfg.image_scene_count, cfg.image_dim,
                                    cfg.image_dim, 3)
            for i in range(cfg.image_scene_count):
                total_img[i, indexes[i, :counts[i], 1],
                          indexes[i, :counts[i],
                                  0]] = sub_values[i, :counts[i]]

            def tone_map(x):
                return x / (x + 1)

            writer.add_image(sub_name,
                             make_grid(tone_map(total_img.movedim(-1, 1))),
                             step)

        step_axis = 1
        total_lighting = values.sum(axis=step_axis)

        save_sub_image('{}/total'.format(name), total_lighting)
        for i in range(cfg.n_steps):
            # index at step_axis
            if is_actual:
                idx = i
            else:
                idx = i - 1
                if idx < 0:
                    continue
            save_sub_image('{}/step_{}'.format(name, i), values[:, idx])

    # get validation data
    validation_data = []
    with torch.no_grad():
        for base_seed in range(which_gpu, num_local_batchs_per_validation,
                               world_size):
            seed = base_seed * world_batch_size + validation_seed_start
            # TODO: if this is too much vram, could keep on cpu until needed
            # (same with imgs)
            data = neural_render_generate_data_full_scene.generate_data(
                batch_size, None, cfg.rays_per_tri, cfg.samples_per_ray,
                cfg.n_steps, seed).to(example_device_tensor)
            validation_data.append(data)

        image_data = neural_render_generate_data_full_scene.generate_data_for_image(
            2**30, cfg.image_scene_count, cfg.image_dim, cfg.samples_per_ray,
            cfg.n_steps, image_seed)
        image_inputs = image_data.standard.inputs.to(example_device_tensor)

        save_image("images/actual", image_data.image_indexes,
                   image_data.standard.values,
                   image_data.standard.inputs.n_samples_per, True)

    for epoch in range(cfg.epochs):
        net.train()

        i = 0

        train_perceptual_loss_tracker = LossTracker(reduce_tensor)
        train_mse_loss_tracker = LossTracker(reduce_tensor)

        steps_since_display = 0
        max_train_step = epoch_size
        format_len = math.floor(math.log10(max_train_step)) + 1

        def get_or_nan(v):
            if v is None:
                return float('nan')
            return v

        def train_display():
            train_mse_loss, nan_count = train_mse_loss_tracker.query_reset()
            train_perceptual_loss, _ = train_perceptual_loss_tracker.query_reset(
            )
            if not disable_all_output:
                print(
                    "{}, epoch {}/{}, step {}/{}, train mse {:.4e}, perc {:4e}, NaN {}"
                    .format(time_for_log(), epoch, cfg.epochs - 1,
                            str((i + 1) * world_batch_size).zfill(format_len),
                            max_train_step, get_or_nan(train_mse_loss),
                            get_or_nan(train_perceptual_loss),
                            nan_count * world_batch_size),
                    flush=True)
                if train_mse_loss is not None:
                    writer.add_scalar("loss/train_mse", train_mse_loss, step)
                if train_perceptual_loss is not None:
                    writer.add_scalar("loss/train_perceptual",
                                      train_perceptual_loss, step)
                writer.add_scalar("lr", lr, step)
                writer.flush()

        steps_since_set_lr = 0

        lr = None

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

            data = neural_render_generate_data_full_scene.generate_data(
                batch_size, None, cfg.rays_per_tri, cfg.samples_per_ray,
                cfg.n_steps, seed).to(example_device_tensor)

            is_first = True

            class NanLoss(Exception):
                pass

            def get_loss():
                nonlocal is_first

                if torch.is_grad_enabled():
                    optimizer.zero_grad()

                outputs = net(data.inputs, cfg.n_steps)
                loss = criterion(outputs, data.values,
                                 data.inputs.total_tri_count)

                if torch.isnan(loss).any():
                    raise NanLoss()

                if is_first:
                    train_perceptual_loss_tracker.update(
                        perceptual(outputs, data.values,
                                   data.inputs.total_tri_count))
                    train_mse_loss_tracker.update(
                        mse(outputs, data.values, data.inputs.total_tri_count))

                if torch.is_grad_enabled():
                    loss.backward()

                    max_norm = norm_avg.x * 1.5
                    this_norm = nn.utils.clip_grad_norm_(
                        net.parameters(), max_norm)
                    norm_avg.update(this_norm)
                    if is_first and not disable_all_output:
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

        test_perceptual_loss_tracker = LossTracker(reduce_tensor)
        test_mse_loss_tracker = LossTracker(reduce_tensor)

        with torch.no_grad():
            for data in validation_data:
                outputs = net(data.inputs, cfg.n_steps)

                test_perceptual_loss_tracker.update(
                    perceptual(outputs, data.values,
                               data.inputs.total_tri_count))
                test_mse_loss_tracker.update(
                    mse(outputs, data.values, data.inputs.total_tri_count))

            save_image("images/output", image_data.image_indexes,
                       net(image_inputs, cfg.n_steps).cpu(),
                       image_data.standard.inputs.n_samples_per, False)

        test_mse_loss, nan_count = test_mse_loss_tracker.query_reset()
        test_perceptual_loss, _ = test_perceptual_loss_tracker.query_reset()
        if not disable_all_output:
            print(
                "{}, epoch {}/{}, lr {:.4e}, test mse {:.4e}, perc {:4e}, NaN {}"
                .format(time_for_log(), epoch, cfg.epochs - 1, lr,
                        get_or_nan(test_mse_loss),
                        get_or_nan(test_perceptual_loss),
                        nan_count * world_batch_size),
                flush=True)

            if test_mse_loss is not None:
                writer.add_scalar("loss/test_mse", test_mse_loss, step)
            if test_perceptual_loss is not None:
                writer.add_scalar("loss/test_perceptual", test_perceptual_loss,
                                  step)

        # if not disable_all_output and (epoch + 1) % cfg.save_model_every == 0:
        #     torch.save(
        #         net, os.path.join(model_save_output, "net_{}.p".format(epoch)))

    # if not disable_all_output:
    #     torch.save(net, os.path.join(model_save_output, "net_final.p"))
    del logger


class Deinit():
    @staticmethod
    def __enter__():
        pass

    @staticmethod
    def __exit__(*_):
        neural_render_generate_data_full_scene.deinit_renderers()


if __name__ == "__main__":
    neural_render_generate_data_full_scene.deinit_renderers()
    with Deinit():
        main()
