#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, compute_rotation_loss, compute_scale_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import time
from utils.visualize_utils import save_tensor_3NN_as_image, save_points_projected_view
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    save_image_bool = False
    save_pcd_image_bool = False
    # loss and psnr result files
    # loss_file_path = os.path.join(dataset.model_path, "loss_results.txt")
    # psnr_file_path = os.path.join(dataset.model_path, "psnr_results.txt")

    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

        # for mask   
        mask = viewpoint_cam.is_masked
        if mask is not None:
            mask = mask.cuda()
            gt_image[mask] = image.detach()[mask]
        
        # for debug, 打印保存原始点云投影到相机画面的图像、渲染图像、GT图像
        # if 0 < iteration <= 300 and iteration % 25 == 0:
        #     print(f"save picture and pointcloud view when iteration {iteration}..... ")
        #     init_pcd = scene.get_init_scene().point_cloud
        #     save_points_projected_view(init_pcd=init_pcd, view_matrix=viewpoint_cam.world_view_transform, proj_matrix=viewpoint_cam.full_proj_transform, image_width=int(viewpoint_cam.image_width),
        #                                 image_height=int(viewpoint_cam.image_height), output_image_path=f"E:/QYF/3DGS/lyf_gaussian-splatting/gaussian-splatting/data/debug/{viewpoint_cam.image_name}_point.jpg")
        #     save_tensor_3NN_as_image(gt_image, f"E:/QYF/3DGS/lyf_gaussian-splatting/gaussian-splatting/data/debug/{viewpoint_cam.image_name}_gt.jpg")
        #     save_tensor_3NN_as_image(image, f"E:/QYF/3DGS/lyf_gaussian-splatting/gaussian-splatting/data/debug/{viewpoint_cam.image_name}_rendered.jpg")
        #     save_pcd_image_bool = True

        
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # 保持为2d高斯和点云法线先验的loss，待调试权重; 目前是通过清零梯度实现，见167行
        # print("原loss: ", loss)
        # rotation_loss = compute_rotation_loss(gaussians._rotation, gaussians._normals)
        # print("旋转loss: ", rotation_loss)
        # scale_loss = compute_scale_loss(gaussians.get_scaling)
        # print("尺度loss: ", scale_loss)
        # lambda_rotation = 0.1
        # lambda_scale = 0.1
        # loss += lambda_rotation * rotation_loss + lambda_scale * scale_loss
        # print("总loss: ", loss)



        # 不透明度loss，鼓励GS不透明度接近于1
        opac_ = gaussians.get_opacity
        loss_opac = (torch.exp(-(opac_)**2 * 5)).mean()
        lambda_opac = 0.01
        loss += lambda_opac * loss_opac

        #新增L2正则化损失，对大尺寸椭球进行惩罚，而不使用硬编码方法

        # max_scale = torch.tensor([0.015, 0.015, 1e-3], device="cuda")  # 设置各轴最大允许尺寸
        # current_scaling = gaussians.get_scaling  
        # exceed_mask = (current_scaling > max_scale).float()  # 超出阈值的部分
        # scale_penalty = torch.mean((current_scaling - max_scale)**2 * exceed_mask) # 仅惩罚超出部分
        # lambda_scale = 0.1
        # loss += lambda_scale * scale_penalty

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
                # image_name = viewpoint_cam.image_name
                # save_loss_and_psnr(iteration, loss, Ll1, loss_opac, image, gt_image, loss_file_path, psnr_file_path, image_name)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step, 优化策略核心实现
            if iteration < opt.iterations:
                # 针对地面部分，位置、旋转、尺寸梯度全部置零
                ground_mask = (gaussians.is_ground == 1 )

                if gaussians._rotation.grad is not None:
                    gaussians._rotation.grad[ground_mask] = 0.0

                if gaussians._scaling.grad is not None:
                    gaussians._scaling.grad[:, 2] = 0.0 # 2DGS，z轴方向不更新
                    gaussians._scaling.grad[ground_mask] = 0.0

                if gaussians._xyz.grad is not None:
                    gaussians._xyz.grad[ground_mask] = 0.0
                    # gaussians._xyz.grad *= 0.0    # 点云足够准确也可将全部GS的位置置零

                
                # BEV任务中，可将地面高斯的不透明度、尺寸强制赋值
                # def inverse_sigmoid(x):
                #     return torch.log(x/(1-x))
                # if gaussians._opacity.grad is not None:
                #     gaussians._opacity.grad[ground_mask] = 0.0
                # gaussians._opacity = inverse_sigmoid(0.999 * torch.ones((gaussians._xyz.shape[0], 1), dtype=torch.float, device="cuda"))
                # gaussians._scaling[:, 2] = torch.log(torch.tensor(1e-3))
                # gaussians._scaling[:, :2] = torch.log(torch.tensor(0.1/2))

                # 确保每个GS的scale不能大于1m，主要为了避免产生过大的高斯。以下的方式可能不够科学，但有用
                # if gaussians._scaling.grad is not None:
                #     # 最大尺寸约束
                #     x_scaling = gaussians.get_scaling[:, 0]
                #     y_scaling = gaussians.get_scaling[:, 1]

                #     # 创建一个布尔掩码，标记出哪些尺度大于 1m
                #     scale_mask_x = x_scaling > 1.0
                #     scale_mask_y = y_scaling > 1.0

                #     # 统计满足条件的高斯椭球数量
                #     # num_exceeding = scale_mask.sum().item()

                #     # 打印或记录日志
                #     # print(f"Number of Gaussians with scaling > 1m: {num_exceeding}")
                #     # 清零对应高斯椭球的梯度
                #     gaussians._scaling[scale_mask_x, 0] = torch.log(torch.tensor(0.99)) # 强制置为0.99
                #     gaussians._scaling[scale_mask_y, 1] = torch.log(torch.tensor(0.99))
                #     gaussians._scaling.grad[scale_mask_x, 0] = 0.0
                #     gaussians._scaling.grad[scale_mask_y, 1] = 0.0

                # update 
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            '''新增截断操作'''
            # with torch.no_grad():
            #     max_scale = torch.tensor([0.1, 0.1, 1e-3], device="cuda") # 设置最大尺寸
            #     current_scaling = gaussians.get_scaling
            #     clamped_scaling = torch.min(current_scaling, max_scale)
            #     gaussians._scaling.data = torch.log(clamped_scaling)
            # # debug
            # assert torch.all(gaussians.get_scaling <= torch.tensor([0.1, 0.1, 1e-3], device="cuda")), "Scaling超出限制"
            '''结束'''

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        # print(f"iteration {iteration} Done.........")
        # time.sleep(0.1)

# def save_loss_and_psnr(iteration, loss, Ll1, loss_opac, image, gt_image, loss_file_path, psnr_file_path, image_name):
#     """
#     保存损失和PSNR到文件
#     """
#     # 保存损失结果
#     with open(loss_file_path, "a") as loss_file:
#         loss_file.write(f"{iteration} {loss.item()} {Ll1.item()} {loss_opac.item()} {image_name}\n")
    
#     # 计算并保存PSNR结果
#     psnr_value = psnr(image, gt_image).mean().item()
#     with open(psnr_file_path, "a") as psnr_file:
#         psnr_file.write(f"{iteration} {psnr_value} {image_name}\n")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
