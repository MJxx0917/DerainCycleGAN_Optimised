import torch
from options import TestOptions
from dataset import dataset_single_test
from model import DerainCycleGAN
from saver import save_imgs
import os
import time


def main():
  # parse options
  parser = TestOptions()
  opts = parser.parse()

  # data loader
  print('\n--- load dataset ---')
  if opts.mode == 0:
    dataset = dataset_single_test(opts, '_rain100H', opts.input_dim_a)
  elif opts.mode == 1:
    dataset = dataset_single_test(opts, '_rain100L', opts.input_dim_b)
  elif opts.mode == 2:
    dataset = dataset_single_test(opts, '_rain12', opts.input_dim_b)
  elif opts.mode == 3:
    dataset = dataset_single_test(opts, '_real', opts.input_dim_b)
  elif opts.mode == 4:
    dataset = dataset_single_test(opts, '_rain800', opts.input_dim_b)
  elif opts.mode == 5:
    dataset = dataset_single_test(opts, '_SPA', opts.input_dim_b)
  elif opts.mode == 6:
    dataset = dataset_single_test(opts, '_practical', opts.input_dim_b)
  elif opts.mode == 7:
    dataset = dataset_single_test(opts, '_real', opts.input_dim_b)
  else:
    raise ValueError("Invalid mode for dataset loading")

  loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opts.nThreads)

  # model
  print('\n--- load model ---')
  model = DerainCycleGAN(opts)
  model.setgpu(opts.gpu)
  model.resume(opts.resume, train=False)
  model.eval()

  # directory
  result_dir = os.path.join(opts.result_dir, opts.name)
  if not os.path.exists(result_dir):
    os.makedirs(result_dir)

  # test
  print('\n--- testing ---')
  total_time = 0
  for idx, (img, needcrop) in enumerate(loader):
    print(f'{idx + 1}/{len(loader)}')
    img = img.cuda()

    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
      output = model.test_forward(img, a2b=opts.a2b)
    end_time = time.time()

    # Compute total time
    total_time += (end_time - start_time)

    # Save images
    save_imgs([output], [f'test_{idx + 1}'], result_dir, needcrop)

  print(f"Total inference time: {total_time:.3f} seconds")
  print(f"Average inference time per image: {total_time / len(loader):.3f} seconds")


if __name__ == '__main__':
  main()

