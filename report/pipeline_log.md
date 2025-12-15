```python
!python pipeline.py --from train --to generate
```

Running pipeline steps: ['train', 'evaluate', 'generate'] (force=True)

============================================================
TRANSFORMER SCALING EXPERIMENT
============================================================

############################################################
# TRANSFORMER - TINY
############################################################

============================================================
Training transformer (tiny)
============================================================
Vocab size: 97
Model parameters: 801,152
Device: cuda
Model device: cuda:0
Batch size: 128
Block size: 256
Total tokens: 397,146,516
Token budget: 397,146,516
Tokens per iter: 32,768
Max iterations: 12,119 (397.1M tokens)
Training:   4% 499/12119 [00:20<07:31, 25.71it/s, loss=1.2947, lr=2.99e-04]
[iter 500] train_loss=1.2788 val_loss=1.2121
Training:   8% 1000/12119 [00:40<07:27, 24.87it/s, loss=1.0705, lr=2.96e-04]
[iter 1000] train_loss=1.0542 val_loss=0.9637
Training:  12% 1498/12119 [01:00<06:51, 25.82it/s, loss=0.9549, lr=2.91e-04]
[iter 1500] train_loss=0.9439 val_loss=0.8515
Training:  16% 1999/12119 [01:21<06:30, 25.89it/s, loss=0.8867, lr=2.84e-04]
[iter 2000] train_loss=0.8734 val_loss=0.7857
Training:  21% 2500/12119 [01:41<06:13, 25.76it/s, loss=0.8299, lr=2.74e-04]
[iter 2500] train_loss=0.8202 val_loss=0.7306
Training:  25% 2998/12119 [02:01<05:53, 25.79it/s, loss=0.7909, lr=2.63e-04]
[iter 3000] train_loss=0.7856 val_loss=0.6957
Training:  29% 3499/12119 [02:21<05:33, 25.82it/s, loss=0.7653, lr=2.50e-04]
[iter 3500] train_loss=0.7640 val_loss=0.6765
Training:  33% 4000/12119 [02:41<05:15, 25.74it/s, loss=0.7381, lr=2.31e-04]
[iter 4000] train_loss=0.7385 val_loss=0.6547
Training:  37% 4498/12119 [03:01<04:55, 25.78it/s, loss=0.7238, lr=2.15e-04]
[iter 4500] train_loss=0.7211 val_loss=0.6374
Training:  41% 4999/12119 [03:22<04:35, 25.81it/s, loss=0.7089, lr=1.98e-04]
[iter 5000] train_loss=0.7081 val_loss=0.6275
Training:  45% 5500/12119 [03:42<04:16, 25.83it/s, loss=0.6974, lr=1.80e-04]
[iter 5500] train_loss=0.6955 val_loss=0.6134
Training:  49% 5998/12119 [04:02<03:57, 25.73it/s, loss=0.6843, lr=1.62e-04]
[iter 6000] train_loss=0.6832 val_loss=0.5998
Training:  54% 6499/12119 [04:22<03:37, 25.85it/s, loss=0.6761, lr=1.44e-04]
[iter 6500] train_loss=0.6700 val_loss=0.5996
Training:  58% 7000/12119 [04:42<03:18, 25.75it/s, loss=0.6671, lr=1.25e-04]
[iter 7000] train_loss=0.6658 val_loss=0.5932
Training:  62% 7498/12119 [05:02<02:59, 25.74it/s, loss=0.6597, lr=1.08e-04]
[iter 7500] train_loss=0.6552 val_loss=0.5874
Training:  66% 7999/12119 [05:23<02:39, 25.76it/s, loss=0.6562, lr=8.67e-05]
[iter 8000] train_loss=0.6575 val_loss=0.5800
Training:  70% 8500/12119 [05:43<02:20, 25.74it/s, loss=0.6488, lr=7.11e-05]
[iter 8500] train_loss=0.6505 val_loss=0.5732
Training:  74% 8998/12119 [06:03<02:00, 25.81it/s, loss=0.6468, lr=5.69e-05]
[iter 9000] train_loss=0.6442 val_loss=0.5806
Training:  78% 9499/12119 [06:23<01:41, 25.75it/s, loss=0.6421, lr=4.42e-05]
[iter 9500] train_loss=0.6410 val_loss=0.5743
Training:  83% 10000/12119 [06:43<01:22, 25.70it/s, loss=0.6387, lr=3.33e-05]
[iter 10000] train_loss=0.6363 val_loss=0.5691
Training:  87% 10498/12119 [07:04<01:03, 25.72it/s, loss=0.6383, lr=2.43e-05]
[iter 10500] train_loss=0.6411 val_loss=0.5660
Training:  91% 10999/12119 [07:24<00:43, 25.75it/s, loss=0.6382, lr=1.74e-05]
[iter 11000] train_loss=0.6347 val_loss=0.5718
Training:  95% 11500/12119 [07:44<00:24, 25.74it/s, loss=0.6386, lr=1.19e-05]
[iter 11500] train_loss=0.6389 val_loss=0.5643
Training:  99% 11998/12119 [08:04<00:04, 25.82it/s, loss=0.6343, lr=1.01e-05]
[iter 12000] train_loss=0.6354 val_loss=0.5657
Training: 100% 12118/12119 [08:09<00:00, 25.86it/s, loss=0.6375, lr=1.00e-05]
[iter 12119] train_loss=0.6369 val_loss=0.5631
Training: 100% 12119/12119 [08:10<00:00, 24.70it/s, loss=0.6375, lr=1.00e-05]
Saved final checkpoint: /content/music-scaling-laws/checkpoints/transformer/transformer_tiny_final.pt
Removed resume checkpoint: /content/music-scaling-laws/checkpoints/transformer/transformer_tiny_last.pt

Results:
  model_type: transformer
  model_size: tiny
  num_params: 801152
  train_loss: 0.6369
  val_loss: 0.5633
  train_time_sec: 490.6870
  tokens_per_sec: 809304.8710
  gpu_memory_mb: 2345.9697

############################################################
# TRANSFORMER - SMALL
############################################################

============================================================
Training transformer (small)
============================================================
Vocab size: 97
Model parameters: 4,176,720
Device: cuda
Model device: cuda:0
Batch size: 128
Block size: 256
Total tokens: 397,146,516
Token budget: 397,146,516
Tokens per iter: 32,768
Max iterations: 12,119 (397.1M tokens)
Training:   4% 500/12119 [00:56<21:52,  8.86it/s, loss=1.0424, lr=2.99e-04]
[iter 500] train_loss=1.0195 val_loss=0.9131
Training:   8% 1000/12119 [01:55<20:57,  8.85it/s, loss=0.8192, lr=2.96e-04]
[iter 1000] train_loss=0.8109 val_loss=0.7294
Training:  12% 1500/12119 [02:53<20:00,  8.85it/s, loss=0.7254, lr=2.91e-04]
[iter 1500] train_loss=0.7178 val_loss=0.6393
Training:  17% 2000/12119 [03:51<19:03,  8.85it/s, loss=0.6560, lr=2.84e-04]
[iter 2000] train_loss=0.6519 val_loss=0.5853
Training:  21% 2500/12119 [04:50<18:04,  8.87it/s, loss=0.6184, lr=2.74e-04]
[iter 2500] train_loss=0.6113 val_loss=0.5602
Training:  25% 3000/12119 [05:48<17:08,  8.87it/s, loss=0.5892, lr=2.63e-04]
[iter 3000] train_loss=0.5826 val_loss=0.5411
Training:  29% 3500/12119 [06:47<16:10,  8.88it/s, loss=0.5713, lr=2.50e-04]
[iter 3500] train_loss=0.5654 val_loss=0.5193
Training:  33% 4000/12119 [07:45<15:16,  8.86it/s, loss=0.5504, lr=2.31e-04]
[iter 4000] train_loss=0.5508 val_loss=0.5165
Training:  37% 4500/12119 [08:44<14:21,  8.84it/s, loss=0.5369, lr=2.15e-04]
[iter 4500] train_loss=0.5372 val_loss=0.5088
Training:  41% 5000/12119 [09:42<13:23,  8.86it/s, loss=0.5271, lr=1.98e-04]
[iter 5000] train_loss=0.5264 val_loss=0.4977
Training:  45% 5500/12119 [10:40<12:24,  8.89it/s, loss=0.5232, lr=1.80e-04]
[iter 5500] train_loss=0.5218 val_loss=0.4986
Training:  50% 6000/12119 [11:39<11:28,  8.89it/s, loss=0.5130, lr=1.62e-04]
[iter 6000] train_loss=0.5145 val_loss=0.4853
Training:  54% 6500/12119 [12:37<10:31,  8.89it/s, loss=0.5077, lr=1.44e-04]
[iter 6500] train_loss=0.5104 val_loss=0.4835
Training:  58% 7000/12119 [13:36<09:37,  8.86it/s, loss=0.5034, lr=1.25e-04]
[iter 7000] train_loss=0.5041 val_loss=0.4791
Training:  62% 7500/12119 [14:34<08:41,  8.85it/s, loss=0.5010, lr=1.08e-04]
[iter 7500] train_loss=0.4983 val_loss=0.4783
Training:  66% 8000/12119 [15:33<07:44,  8.88it/s, loss=0.4953, lr=8.67e-05]
[iter 8000] train_loss=0.4940 val_loss=0.4738
Training:  70% 8500/12119 [16:31<06:47,  8.88it/s, loss=0.4957, lr=7.11e-05]
[iter 8500] train_loss=0.4950 val_loss=0.4694
Training:  74% 9000/12119 [17:29<05:52,  8.85it/s, loss=0.4890, lr=5.69e-05]
[iter 9000] train_loss=0.4889 val_loss=0.4687
Training:  78% 9500/12119 [18:28<04:54,  8.89it/s, loss=0.4864, lr=4.42e-05]
[iter 9500] train_loss=0.4888 val_loss=0.4671
Training:  83% 10000/12119 [19:26<03:58,  8.87it/s, loss=0.4877, lr=3.33e-05]
[iter 10000] train_loss=0.4850 val_loss=0.4617
Training:  87% 10500/12119 [20:25<03:02,  8.87it/s, loss=0.4814, lr=2.43e-05]
[iter 10500] train_loss=0.4826 val_loss=0.4625
Training:  91% 11000/12119 [21:23<02:06,  8.87it/s, loss=0.4785, lr=1.74e-05]
[iter 11000] train_loss=0.4826 val_loss=0.4636
Training:  95% 11500/12119 [22:21<01:09,  8.88it/s, loss=0.4815, lr=1.19e-05]
[iter 11500] train_loss=0.4826 val_loss=0.4624
Training:  99% 12000/12119 [23:20<00:13,  8.89it/s, loss=0.4757, lr=1.01e-05]
[iter 12000] train_loss=0.4756 val_loss=0.4663
Training: 100% 12119/12119 [23:35<00:00,  8.87it/s, loss=0.4795, lr=1.00e-05]
[iter 12119] train_loss=0.4794 val_loss=0.4610
Training: 100% 12119/12119 [23:37<00:00,  8.55it/s, loss=0.4795, lr=1.00e-05]
Saved final checkpoint: /content/music-scaling-laws/checkpoints/transformer/transformer_small_final.pt
Removed resume checkpoint: /content/music-scaling-laws/checkpoints/transformer/transformer_small_last.pt

Results:
  model_type: transformer
  model_size: small
  num_params: 4176720
  train_loss: 0.4794
  val_loss: 0.4595
  train_time_sec: 1417.3932
  tokens_per_sec: 280173.0517
  gpu_memory_mb: 5841.1479

############################################################
# TRANSFORMER - MEDIUM
############################################################

============================================================
Training transformer (medium)
============================================================
Vocab size: 97
Model parameters: 18,643,240
Device: cuda
Model device: cuda:0
Batch size: 128
Block size: 256
Total tokens: 397,146,516
Token budget: 397,146,516
Tokens per iter: 32,768
Max iterations: 12,119 (397.1M tokens)
Training:   4% 500/12119 [02:45<1:04:04,  3.02it/s, loss=0.8563, lr=2.99e-04]
[iter 500] train_loss=0.8387 val_loss=0.7467
Training:   8% 1000/12119 [05:36<1:01:19,  3.02it/s, loss=0.6631, lr=2.96e-04]
[iter 1000] train_loss=0.6515 val_loss=0.5906
Training:  12% 1500/12119 [08:27<58:31,  3.02it/s, loss=0.5872, lr=2.91e-04]
[iter 1500] train_loss=0.5814 val_loss=0.5348
Training:  17% 2000/12119 [11:19<55:47,  3.02it/s, loss=0.5488, lr=2.84e-04]
[iter 2000] train_loss=0.5449 val_loss=0.5108
Training:  21% 2500/12119 [14:10<53:00,  3.02it/s, loss=0.5238, lr=2.74e-04]
[iter 2500] train_loss=0.5163 val_loss=0.4986
Training:  25% 3000/12119 [17:01<50:14,  3.02it/s, loss=0.5041, lr=2.63e-04]
[iter 3000] train_loss=0.5034 val_loss=0.4794
Training:  29% 3500/12119 [19:52<47:30,  3.02it/s, loss=0.4916, lr=2.50e-04]
[iter 3500] train_loss=0.4897 val_loss=0.4741
Training:  33% 4000/12119 [22:43<45:21,  2.98it/s, loss=0.4812, lr=2.31e-04]
[iter 4000] train_loss=0.4816 val_loss=0.4636
Training:  37% 4500/12119 [25:35<41:56,  3.03it/s, loss=0.4755, lr=2.15e-04]
[iter 4500] train_loss=0.4745 val_loss=0.4576
Training:  41% 5000/12119 [28:26<39:13,  3.03it/s, loss=0.4678, lr=1.98e-04]
[iter 5000] train_loss=0.4681 val_loss=0.4515
Training:  45% 5500/12119 [31:17<36:28,  3.02it/s, loss=0.4628, lr=1.80e-04]
[iter 5500] train_loss=0.4600 val_loss=0.4534
Training:  50% 6000/12119 [34:08<33:43,  3.02it/s, loss=0.4583, lr=1.62e-04]
[iter 6000] train_loss=0.4583 val_loss=0.4465
Training:  54% 6500/12119 [37:00<30:57,  3.02it/s, loss=0.4568, lr=1.44e-04]
[iter 6500] train_loss=0.4571 val_loss=0.4437
Training:  58% 7000/12119 [39:51<28:12,  3.02it/s, loss=0.4510, lr=1.25e-04]
[iter 7000] train_loss=0.4539 val_loss=0.4412
Training:  62% 7500/12119 [42:42<25:27,  3.02it/s, loss=0.4480, lr=1.08e-04]
[iter 7500] train_loss=0.4480 val_loss=0.4352
Training:  66% 8000/12119 [45:33<22:41,  3.02it/s, loss=0.4442, lr=8.67e-05]
[iter 8000] train_loss=0.4454 val_loss=0.4406
Training:  70% 8500/12119 [48:24<19:56,  3.02it/s, loss=0.4410, lr=7.11e-05]
[iter 8500] train_loss=0.4395 val_loss=0.4305
Training:  74% 9000/12119 [51:15<17:11,  3.02it/s, loss=0.4405, lr=5.69e-05]
[iter 9000] train_loss=0.4406 val_loss=0.4351
Training:  78% 9500/12119 [54:06<14:26,  3.02it/s, loss=0.4395, lr=4.42e-05]
[iter 9500] train_loss=0.4420 val_loss=0.4311
Training:  83% 10000/12119 [56:58<11:40,  3.02it/s, loss=0.4378, lr=3.33e-05]
[iter 10000] train_loss=0.4378 val_loss=0.4310
Training:  87% 10500/12119 [59:49<08:55,  3.02it/s, loss=0.4374, lr=2.43e-05]
[iter 10500] train_loss=0.4369 val_loss=0.4310
Training:  91% 11000/12119 [1:02:40<06:10,  3.02it/s, loss=0.4348, lr=1.74e-05]
[iter 11000] train_loss=0.4354 val_loss=0.4236
Training:  95% 11500/12119 [1:05:31<03:24,  3.02it/s, loss=0.4357, lr=1.19e-05]
[iter 11500] train_loss=0.4352 val_loss=0.4271
Training:  99% 12000/12119 [1:08:22<00:39,  3.02it/s, loss=0.4332, lr=1.01e-05]
[iter 12000] train_loss=0.4335 val_loss=0.4272
Training: 100% 12119/12119 [1:09:07<00:00,  3.02it/s, loss=0.4373, lr=1.00e-05]
[iter 12119] train_loss=0.4362 val_loss=0.4268
Training: 100% 12119/12119 [1:09:13<00:00,  2.92it/s, loss=0.4373, lr=1.00e-05]
Saved final checkpoint: /content/music-scaling-laws/checkpoints/transformer/transformer_medium_final.pt
Removed resume checkpoint: /content/music-scaling-laws/checkpoints/transformer/transformer_medium_last.pt

Results:
  model_type: transformer
  model_size: medium
  num_params: 18643240
  train_loss: 0.4362
  val_loss: 0.4280
  train_time_sec: 4153.2904
  tokens_per_sec: 95614.6455
  gpu_memory_mb: 12507.4648

############################################################
# TRANSFORMER - LARGE
############################################################

============================================================
Training transformer (large)
============================================================
Vocab size: 97
Model parameters: 55,444,740
Device: cuda
Model device: cuda:0
Batch size: 128
Block size: 256
Total tokens: 397,146,516
Token budget: 397,146,516
Tokens per iter: 32,768
Max iterations: 12,119 (397.1M tokens)
Training:   4% 500/12119 [06:55<2:40:22,  1.21it/s, loss=0.7947, lr=2.99e-04]
[iter 500] train_loss=0.7761 val_loss=0.6999
Training:   8% 1000/12119 [14:04<2:33:26,  1.21it/s, loss=0.5986, lr=2.96e-04]
[iter 1000] train_loss=0.5883 val_loss=0.5567
Training:  12% 1500/12119 [21:13<2:26:42,  1.21it/s, loss=0.5324, lr=2.91e-04]
[iter 1500] train_loss=0.5270 val_loss=0.5052
Training:  17% 2000/12119 [28:22<2:19:39,  1.21it/s, loss=0.5044, lr=2.84e-04]
[iter 2000] train_loss=0.5019 val_loss=0.4858
Training:  21% 2500/12119 [35:32<2:12:45,  1.21it/s, loss=0.4869, lr=2.74e-04]
[iter 2500] train_loss=0.4854 val_loss=0.4735
Training:  25% 3000/12119 [42:41<2:05:51,  1.21it/s, loss=0.4730, lr=2.63e-04]
[iter 3000] train_loss=0.4744 val_loss=0.4605
Training:  29% 3500/12119 [49:50<1:59:04,  1.21it/s, loss=0.4661, lr=2.50e-04]
[iter 3500] train_loss=0.4604 val_loss=0.4515
Training:  33% 4000/12119 [56:59<1:52:05,  1.21it/s, loss=0.4575, lr=2.31e-04]
[iter 4000] train_loss=0.4553 val_loss=0.4476
Training:  37% 4500/12119 [1:04:09<1:45:11,  1.21it/s, loss=0.4507, lr=2.15e-04]
[iter 4500] train_loss=0.4504 val_loss=0.4418
Training:  41% 5000/12119 [1:11:18<1:38:19,  1.21it/s, loss=0.4464, lr=1.98e-04]
[iter 5000] train_loss=0.4458 val_loss=0.4376
Training:  45% 5500/12119 [1:18:27<1:31:26,  1.21it/s, loss=0.4426, lr=1.80e-04]
[iter 5500] train_loss=0.4400 val_loss=0.4362
Training:  50% 6000/12119 [1:25:37<1:24:31,  1.21it/s, loss=0.4372, lr=1.62e-04]
[iter 6000] train_loss=0.4375 val_loss=0.4297
Training:  54% 6500/12119 [1:32:47<1:17:33,  1.21it/s, loss=0.4336, lr=1.44e-04]
[iter 6500] train_loss=0.4328 val_loss=0.4291
Training:  58% 7000/12119 [1:39:57<1:10:42,  1.21it/s, loss=0.4292, lr=1.25e-04]
[iter 7000] train_loss=0.4318 val_loss=0.4259
Training:  62% 7500/12119 [1:47:07<1:03:47,  1.21it/s, loss=0.4297, lr=1.08e-04]
[iter 7500] train_loss=0.4288 val_loss=0.4242
Training:  66% 8000/12119 [1:54:16<56:53,  1.21it/s, loss=0.4240, lr=8.67e-05]
[iter 8000] train_loss=0.4238 val_loss=0.4227
Training:  70% 8500/12119 [2:01:26<49:59,  1.21it/s, loss=0.4222, lr=7.11e-05]
[iter 8500] train_loss=0.4196 val_loss=0.4212
Training:  74% 9000/12119 [2:08:35<43:02,  1.21it/s, loss=0.4215, lr=5.69e-05]
[iter 9000] train_loss=0.4202 val_loss=0.4172
Training:  78% 9500/12119 [2:15:44<36:08,  1.21it/s, loss=0.4188, lr=4.42e-05]
[iter 9500] train_loss=0.4177 val_loss=0.4183
Training:  83% 10000/12119 [2:22:53<29:14,  1.21it/s, loss=0.4181, lr=3.33e-05]
[iter 10000] train_loss=0.4146 val_loss=0.4188
Training:  87% 10500/12119 [2:30:04<22:20,  1.21it/s, loss=0.4156, lr=2.43e-05]
[iter 10500] train_loss=0.4148 val_loss=0.4163
Training:  91% 11000/12119 [2:37:13<15:26,  1.21it/s, loss=0.4133, lr=1.74e-05]
[iter 11000] train_loss=0.4171 val_loss=0.4135
Training:  95% 11500/12119 [2:44:21<08:32,  1.21it/s, loss=0.4126, lr=1.19e-05]
[iter 11500] train_loss=0.4135 val_loss=0.4081
Training:  99% 12000/12119 [2:51:31<01:38,  1.21it/s, loss=0.4125, lr=1.01e-05]
[iter 12000] train_loss=0.4099 val_loss=0.4144
Training: 100% 12119/12119 [2:53:24<00:00,  1.21it/s, loss=0.4139, lr=1.00e-05]
[iter 12119] train_loss=0.4146 val_loss=0.4165
Training: 100% 12119/12119 [2:53:38<00:00,  1.16it/s, loss=0.4139, lr=1.00e-05]
Saved final checkpoint: /content/music-scaling-laws/checkpoints/transformer/transformer_large_final.pt
Removed resume checkpoint: /content/music-scaling-laws/checkpoints/transformer/transformer_large_last.pt

Results:
  model_type: transformer
  model_size: large
  num_params: 55444740
  train_loss: 0.4146
  val_loss: 0.4132
  train_time_sec: 10418.1446
  tokens_per_sec: 38117.6696
  gpu_memory_mb: 25162.8076

############################################################
# TRANSFORMER - XL
############################################################
[override] transformer/xl max_tokens set to full epoch: 397,146,516

============================================================
Training transformer (xl)
============================================================
Vocab size: 97
Model parameters: 99,209,472
Device: cuda
Model device: cuda:0
Batch size: 128
Block size: 256
Total tokens: 397,146,516
Token budget: 397,146,516
Tokens per iter: 32,768
Max iterations: 12,119 (397.1M tokens)
Training:   4% 500/12119 [11:16<4:20:58,  1.35s/it, loss=0.7638, lr=2.99e-04]
[iter 500] train_loss=0.7466 val_loss=0.6608
Training:   8% 1000/12119 [22:55<4:09:58,  1.35s/it, loss=0.5708, lr=2.96e-04]
[iter 1000] train_loss=0.5621 val_loss=0.5302
Training:  12% 1500/12119 [34:33<3:58:44,  1.35s/it, loss=0.5191, lr=2.91e-04]
[iter 1500] train_loss=0.5140 val_loss=0.4963
Training:  17% 2000/12119 [46:11<3:47:17,  1.35s/it, loss=0.4892, lr=2.84e-04]
[iter 2000] train_loss=0.4856 val_loss=0.4685
Training:  21% 2500/12119 [57:51<3:36:03,  1.35s/it, loss=0.4719, lr=2.74e-04]
[iter 2500] train_loss=0.4719 val_loss=0.4606
Training:  25% 3000/12119 [1:09:30<3:25:00,  1.35s/it, loss=0.4634, lr=2.63e-04]
[iter 3000] train_loss=0.4619 val_loss=0.4527
Training:  29% 3500/12119 [1:21:08<3:13:44,  1.35s/it, loss=0.4548, lr=2.50e-04]
[iter 3500] train_loss=0.4513 val_loss=0.4493
Training:  33% 4000/12119 [1:32:46<3:02:21,  1.35s/it, loss=0.4497, lr=2.31e-04]
[iter 4000] train_loss=0.4499 val_loss=0.4407
Training:  37% 4500/12119 [1:44:26<2:51:05,  1.35s/it, loss=0.4380, lr=2.15e-04]
[iter 4500] train_loss=0.4379 val_loss=0.4364
Training:  41% 5000/12119 [1:56:05<2:39:49,  1.35s/it, loss=0.4373, lr=1.98e-04]
[iter 5000] train_loss=0.4362 val_loss=0.4306
Training:  45% 5500/12119 [2:07:42<2:28:39,  1.35s/it, loss=0.4321, lr=1.80e-04]
[iter 5500] train_loss=0.4297 val_loss=0.4311
Training:  50% 6000/12119 [2:19:21<2:17:33,  1.35s/it, loss=0.4295, lr=1.62e-04]
[iter 6000] train_loss=0.4273 val_loss=0.4230
Training:  54% 6500/12119 [2:31:01<2:06:20,  1.35s/it, loss=0.4282, lr=1.44e-04]
[iter 6500] train_loss=0.4221 val_loss=0.4225
Training:  58% 7000/12119 [2:42:40<1:55:05,  1.35s/it, loss=0.4213, lr=1.25e-04]
[iter 7000] train_loss=0.4220 val_loss=0.4237
Training:  62% 7500/12119 [2:54:18<1:43:50,  1.35s/it, loss=0.4180, lr=1.08e-04]
[iter 7500] train_loss=0.4212 val_loss=0.4154
Training:  66% 8000/12119 [3:05:57<1:32:31,  1.35s/it, loss=0.4159, lr=8.67e-05]
[iter 8000] train_loss=0.4156 val_loss=0.4159
Training:  70% 8500/12119 [3:17:36<1:21:22,  1.35s/it, loss=0.4139, lr=7.11e-05]
[iter 8500] train_loss=0.4142 val_loss=0.4154
Training:  74% 9000/12119 [3:29:14<1:10:03,  1.35s/it, loss=0.4104, lr=5.69e-05]
[iter 9000] train_loss=0.4110 val_loss=0.4127
Training:  78% 9500/12119 [3:40:52<58:49,  1.35s/it, loss=0.4111, lr=4.42e-05]
[iter 9500] train_loss=0.4112 val_loss=0.4112
Training:  83% 10000/12119 [3:52:30<47:35,  1.35s/it, loss=0.4088, lr=3.33e-05]
[iter 10000] train_loss=0.4099 val_loss=0.4078
Training:  87% 10500/12119 [4:04:10<36:23,  1.35s/it, loss=0.4057, lr=2.43e-05]
[iter 10500] train_loss=0.4095 val_loss=0.4052
Training:  91% 11000/12119 [4:15:48<25:07,  1.35s/it, loss=0.4089, lr=1.74e-05]
[iter 11000] train_loss=0.4072 val_loss=0.4075
Training:  95% 11500/12119 [4:27:26<13:53,  1.35s/it, loss=0.4038, lr=1.19e-05]
[iter 11500] train_loss=0.4044 val_loss=0.4078
Training:  99% 12000/12119 [4:39:04<02:40,  1.35s/it, loss=0.4050, lr=1.01e-05]
[iter 12000] train_loss=0.4057 val_loss=0.4076
Training: 100% 12119/12119 [4:42:09<00:00,  1.35s/it, loss=0.4066, lr=1.00e-05]
[iter 12119] train_loss=0.4050 val_loss=0.4058
Training: 100% 12119/12119 [4:42:31<00:00,  1.40s/it, loss=0.4066, lr=1.00e-05]
Saved final checkpoint: /content/music-scaling-laws/checkpoints/transformer/transformer_xl_final.pt
Removed resume checkpoint: /content/music-scaling-laws/checkpoints/transformer/transformer_xl_last.pt

Results:
  model_type: transformer
  model_size: xl
  num_params: 99209472
  train_loss: 0.4050
  val_loss: 0.4088
  train_time_sec: 16951.6988
  tokens_per_sec: 23426.2888
  gpu_memory_mb: 36183.0698

============================================================
LSTM SCALING EXPERIMENT
============================================================

############################################################
# LSTM - TINY
############################################################

============================================================
Training lstm (tiny)
============================================================
Vocab size: 97
Model parameters: 925,921
Device: cuda
Model device: cuda:0
Batch size: 128
Block size: 256
Total tokens: 397,146,516
Token budget: 397,146,516
Tokens per iter: 32,768
Max iterations: 12,119 (397.1M tokens)
Training:   4% 496/12119 [00:07<02:59, 64.84it/s, loss=1.1512, lr=2.99e-04]
[iter 500] train_loss=1.1414 val_loss=1.1073
Training:   8% 1000/12119 [00:16<02:50, 65.12it/s, loss=1.0324, lr=2.96e-04]
[iter 1000] train_loss=1.0253 val_loss=1.0022
Training:  12% 1497/12119 [00:24<02:43, 64.99it/s, loss=0.9661, lr=2.91e-04]
[iter 1500] train_loss=0.9559 val_loss=0.9321
Training:  16% 1994/12119 [00:32<02:35, 64.92it/s, loss=0.9113, lr=2.84e-04]
[iter 2000] train_loss=0.9067 val_loss=0.8863
Training:  21% 2498/12119 [00:40<02:28, 64.72it/s, loss=0.8714, lr=2.74e-04]
[iter 2500] train_loss=0.8645 val_loss=0.8514
Training:  25% 2995/12119 [00:48<02:20, 65.03it/s, loss=0.8385, lr=2.63e-04]
[iter 3000] train_loss=0.8274 val_loss=0.8108
Training:  29% 3499/12119 [00:56<02:14, 64.17it/s, loss=0.8034, lr=2.50e-04]
[iter 3500] train_loss=0.8026 val_loss=0.7868
Training:  33% 3996/12119 [01:04<02:04, 65.08it/s, loss=0.7761, lr=2.31e-04]
[iter 4000] train_loss=0.7748 val_loss=0.7564
Training:  37% 4500/12119 [01:13<02:01, 62.57it/s, loss=0.7591, lr=2.15e-04]
[iter 4500] train_loss=0.7582 val_loss=0.7438
Training:  41% 4997/12119 [01:21<01:49, 65.03it/s, loss=0.7412, lr=1.98e-04]
[iter 5000] train_loss=0.7405 val_loss=0.7275
Training:  45% 5494/12119 [01:29<01:41, 65.10it/s, loss=0.7312, lr=1.80e-04]
[iter 5500] train_loss=0.7255 val_loss=0.7125
Training:  49% 5998/12119 [01:37<01:34, 64.90it/s, loss=0.7209, lr=1.62e-04]
[iter 6000] train_loss=0.7136 val_loss=0.7000
Training:  54% 6495/12119 [01:45<01:26, 64.87it/s, loss=0.7077, lr=1.44e-04]
[iter 6500] train_loss=0.7044 val_loss=0.6911
Training:  58% 6999/12119 [01:53<01:18, 65.00it/s, loss=0.6987, lr=1.25e-04]
[iter 7000] train_loss=0.6992 val_loss=0.6900
Training:  62% 7496/12119 [02:01<01:11, 65.11it/s, loss=0.6957, lr=1.08e-04]
[iter 7500] train_loss=0.6924 val_loss=0.6866
Training:  66% 8000/12119 [02:10<01:03, 64.68it/s, loss=0.6894, lr=8.67e-05]
[iter 8000] train_loss=0.6880 val_loss=0.6823
Training:  70% 8497/12119 [02:18<00:55, 64.71it/s, loss=0.6842, lr=7.11e-05]
[iter 8500] train_loss=0.6839 val_loss=0.6746
Training:  74% 8994/12119 [02:26<00:48, 64.63it/s, loss=0.6822, lr=5.69e-05]
[iter 9000] train_loss=0.6833 val_loss=0.6788
Training:  78% 9498/12119 [02:34<00:40, 64.72it/s, loss=0.6817, lr=4.42e-05]
[iter 9500] train_loss=0.6790 val_loss=0.6732
Training:  82% 9994/12119 [02:42<00:32, 64.99it/s, loss=0.6755, lr=3.33e-05]
[iter 10000] train_loss=0.6739 val_loss=0.6661
Training:  87% 10498/12119 [02:50<00:24, 64.88it/s, loss=0.6739, lr=2.43e-05]
[iter 10500] train_loss=0.6767 val_loss=0.6589
Training:  91% 10995/12119 [02:59<00:17, 64.98it/s, loss=0.6720, lr=1.74e-05]
[iter 11000] train_loss=0.6733 val_loss=0.6665
Training:  95% 11499/12119 [03:07<00:09, 64.95it/s, loss=0.6720, lr=1.19e-05]
[iter 11500] train_loss=0.6716 val_loss=0.6675
Training:  99% 11996/12119 [03:15<00:01, 65.00it/s, loss=0.6737, lr=1.01e-05]
[iter 12000] train_loss=0.6762 val_loss=0.6606
Training: 100% 12115/12119 [03:17<00:00, 64.65it/s, loss=0.6736, lr=1.00e-05]
[iter 12119] train_loss=0.6739 val_loss=0.6658
Training: 100% 12119/12119 [03:18<00:00, 61.18it/s, loss=0.6736, lr=1.00e-05]
Saved final checkpoint: /content/music-scaling-laws/checkpoints/lstm/lstm_tiny_final.pt
Removed resume checkpoint: /content/music-scaling-laws/checkpoints/lstm/lstm_tiny_last.pt

Results:
  model_type: lstm
  model_size: tiny
  num_params: 925921
  train_loss: 0.6739
  val_loss: 0.6650
  train_time_sec: 198.0751
  tokens_per_sec: 2004873.2793
  gpu_memory_mb: 36183.0698

############################################################
# LSTM - SMALL
############################################################

============================================================
Training lstm (small)
============================================================
Vocab size: 97
Model parameters: 3,728,993
Device: cuda
Model device: cuda:0
Batch size: 128
Block size: 256
Total tokens: 397,146,516
Token budget: 397,146,516
Tokens per iter: 32,768
Max iterations: 12,119 (397.1M tokens)
Training:   4% 498/12119 [00:14<05:22, 36.02it/s, loss=1.0783, lr=2.99e-04]
[iter 500] train_loss=1.0676 val_loss=1.0285
Training:   8% 998/12119 [00:28<05:08, 36.10it/s, loss=0.9437, lr=2.96e-04]
[iter 1000] train_loss=0.9350 val_loss=0.9083
Training:  12% 1498/12119 [00:43<04:59, 35.42it/s, loss=0.8570, lr=2.91e-04]
[iter 1500] train_loss=0.8465 val_loss=0.8230
Training:  16% 1998/12119 [00:57<04:40, 36.12it/s, loss=0.7880, lr=2.84e-04]
[iter 2000] train_loss=0.7815 val_loss=0.7601
Training:  21% 2498/12119 [01:12<04:27, 36.03it/s, loss=0.7410, lr=2.74e-04]
[iter 2500] train_loss=0.7308 val_loss=0.7160
Training:  25% 2998/12119 [01:26<04:12, 36.10it/s, loss=0.7078, lr=2.63e-04]
[iter 3000] train_loss=0.6980 val_loss=0.6854
Training:  29% 3498/12119 [01:40<03:59, 36.04it/s, loss=0.6805, lr=2.50e-04]
[iter 3500] train_loss=0.6749 val_loss=0.6639
Training:  33% 3998/12119 [01:55<03:45, 35.97it/s, loss=0.6631, lr=2.31e-04]
[iter 4000] train_loss=0.6645 val_loss=0.6447
Training:  37% 4498/12119 [02:10<03:31, 36.10it/s, loss=0.6479, lr=2.15e-04]
[iter 4500] train_loss=0.6466 val_loss=0.6319
Training:  41% 4998/12119 [02:24<03:17, 36.13it/s, loss=0.6375, lr=1.98e-04]
[iter 5000] train_loss=0.6369 val_loss=0.6196
Training:  45% 5498/12119 [02:39<03:03, 36.13it/s, loss=0.6297, lr=1.80e-04]
[iter 5500] train_loss=0.6261 val_loss=0.6121
Training:  49% 5998/12119 [02:53<02:49, 36.19it/s, loss=0.6162, lr=1.62e-04]
[iter 6000] train_loss=0.6132 val_loss=0.6029
Training:  54% 6498/12119 [03:08<02:35, 36.14it/s, loss=0.6093, lr=1.44e-04]
[iter 6500] train_loss=0.6083 val_loss=0.6008
Training:  58% 6998/12119 [03:22<02:21, 36.16it/s, loss=0.6006, lr=1.25e-04]
[iter 7000] train_loss=0.6042 val_loss=0.5922
Training:  62% 7498/12119 [03:36<02:07, 36.16it/s, loss=0.5996, lr=1.08e-04]
[iter 7500] train_loss=0.5996 val_loss=0.5912
Training:  66% 7998/12119 [03:51<01:53, 36.20it/s, loss=0.5947, lr=8.67e-05]
[iter 8000] train_loss=0.5923 val_loss=0.5862
Training:  70% 8498/12119 [04:05<01:39, 36.22it/s, loss=0.5888, lr=7.11e-05]
[iter 8500] train_loss=0.5883 val_loss=0.5800
Training:  74% 8998/12119 [04:20<01:26, 36.20it/s, loss=0.5873, lr=5.69e-05]
[iter 9000] train_loss=0.5877 val_loss=0.5771
Training:  78% 9498/12119 [04:34<01:12, 36.18it/s, loss=0.5820, lr=4.42e-05]
[iter 9500] train_loss=0.5847 val_loss=0.5752
Training:  82% 9998/12119 [04:49<00:58, 36.24it/s, loss=0.5835, lr=3.33e-05]
[iter 10000] train_loss=0.5835 val_loss=0.5716
Training:  87% 10498/12119 [05:03<00:44, 36.20it/s, loss=0.5815, lr=2.43e-05]
[iter 10500] train_loss=0.5829 val_loss=0.5722
Training:  91% 10998/12119 [05:18<00:31, 36.15it/s, loss=0.5791, lr=1.74e-05]
[iter 11000] train_loss=0.5786 val_loss=0.5715
Training:  95% 11498/12119 [05:32<00:17, 36.03it/s, loss=0.5809, lr=1.19e-05]
[iter 11500] train_loss=0.5806 val_loss=0.5711
Training:  99% 11998/12119 [05:47<00:03, 36.04it/s, loss=0.5789, lr=1.01e-05]
[iter 12000] train_loss=0.5777 val_loss=0.5726
Training: 100% 12118/12119 [05:51<00:00, 36.10it/s, loss=0.5807, lr=1.00e-05]
[iter 12119] train_loss=0.5809 val_loss=0.5723
Training: 100% 12119/12119 [05:51<00:00, 34.46it/s, loss=0.5807, lr=1.00e-05]
Saved final checkpoint: /content/music-scaling-laws/checkpoints/lstm/lstm_small_final.pt
Removed resume checkpoint: /content/music-scaling-laws/checkpoints/lstm/lstm_small_last.pt

Results:
  model_type: lstm
  model_size: small
  num_params: 3728993
  train_loss: 0.5809
  val_loss: 0.5714
  train_time_sec: 351.7175
  tokens_per_sec: 1129074.7348
  gpu_memory_mb: 36183.0698

############################################################
# LSTM - MEDIUM
############################################################

============================================================
Training lstm (medium)
============================================================
Vocab size: 97
Model parameters: 23,194,721
Device: cuda
Model device: cuda:0
Batch size: 128
Block size: 256
Total tokens: 397,146,516
Token budget: 397,146,516
Tokens per iter: 32,768
Max iterations: 12,119 (397.1M tokens)
Training:   4% 499/12119 [00:38<14:54, 12.99it/s, loss=1.0134, lr=2.99e-04]
[iter 500] train_loss=1.0006 val_loss=0.9532
Training:   8% 999/12119 [01:19<14:20, 12.93it/s, loss=0.8241, lr=2.96e-04]
[iter 1000] train_loss=0.8122 val_loss=0.7837
Training:  12% 1499/12119 [01:59<13:35, 13.03it/s, loss=0.7172, lr=2.91e-04]
[iter 1500] train_loss=0.7033 val_loss=0.6893
Training:  16% 1999/12119 [02:40<12:59, 12.98it/s, loss=0.6605, lr=2.84e-04]
[iter 2000] train_loss=0.6546 val_loss=0.6355
Training:  21% 2499/12119 [03:21<12:21, 12.97it/s, loss=0.6231, lr=2.74e-04]
[iter 2500] train_loss=0.6143 val_loss=0.6109
Training:  25% 2999/12119 [04:01<11:43, 12.96it/s, loss=0.5979, lr=2.63e-04]
[iter 3000] train_loss=0.5862 val_loss=0.5778
Training:  29% 3499/12119 [04:42<11:04, 12.98it/s, loss=0.5690, lr=2.50e-04]
[iter 3500] train_loss=0.5699 val_loss=0.5632
Training:  33% 3999/12119 [05:22<10:28, 12.93it/s, loss=0.5519, lr=2.31e-04]
[iter 4000] train_loss=0.5521 val_loss=0.5448
Training:  37% 4499/12119 [06:03<09:47, 12.96it/s, loss=0.5419, lr=2.15e-04]
[iter 4500] train_loss=0.5396 val_loss=0.5342
Training:  41% 4999/12119 [06:44<09:09, 12.97it/s, loss=0.5325, lr=1.98e-04]
[iter 5000] train_loss=0.5303 val_loss=0.5233
Training:  45% 5499/12119 [07:24<08:31, 12.95it/s, loss=0.5240, lr=1.80e-04]
[iter 5500] train_loss=0.5219 val_loss=0.5193
Training:  50% 5999/12119 [08:05<07:53, 12.93it/s, loss=0.5183, lr=1.62e-04]
[iter 6000] train_loss=0.5150 val_loss=0.5129
Training:  54% 6499/12119 [08:46<07:13, 12.97it/s, loss=0.5147, lr=1.44e-04]
[iter 6500] train_loss=0.5085 val_loss=0.5073
Training:  58% 6999/12119 [09:27<06:32, 13.03it/s, loss=0.5042, lr=1.25e-04]
[iter 7000] train_loss=0.5021 val_loss=0.5018
Training:  62% 7499/12119 [10:07<05:54, 13.02it/s, loss=0.5030, lr=1.08e-04]
[iter 7500] train_loss=0.5035 val_loss=0.4967
Training:  66% 7999/12119 [10:48<05:18, 12.95it/s, loss=0.4949, lr=8.67e-05]
[iter 8000] train_loss=0.4953 val_loss=0.4972
Training:  70% 8499/12119 [11:29<04:39, 12.96it/s, loss=0.4946, lr=7.11e-05]
[iter 8500] train_loss=0.4934 val_loss=0.5026
Training:  74% 8999/12119 [12:09<04:00, 12.96it/s, loss=0.4934, lr=5.69e-05]
[iter 9000] train_loss=0.4930 val_loss=0.4867
Training:  78% 9499/12119 [12:50<03:21, 13.00it/s, loss=0.4917, lr=4.42e-05]
[iter 9500] train_loss=0.4906 val_loss=0.4888
Training:  83% 9999/12119 [13:30<02:44, 12.91it/s, loss=0.4884, lr=3.33e-05]
[iter 10000] train_loss=0.4908 val_loss=0.4865
Training:  87% 10499/12119 [14:11<02:05, 12.96it/s, loss=0.4877, lr=2.43e-05]
[iter 10500] train_loss=0.4874 val_loss=0.4884
Training:  91% 10999/12119 [14:52<01:26, 12.93it/s, loss=0.4853, lr=1.74e-05]
[iter 11000] train_loss=0.4853 val_loss=0.4832
Training:  95% 11499/12119 [15:32<00:47, 13.02it/s, loss=0.4858, lr=1.19e-05]
[iter 11500] train_loss=0.4849 val_loss=0.4827
Training:  99% 11999/12119 [16:13<00:09, 12.99it/s, loss=0.4853, lr=1.01e-05]
[iter 12000] train_loss=0.4844 val_loss=0.4863
Training: 100% 12119/12119 [16:24<00:00, 12.98it/s, loss=0.4845, lr=1.00e-05]
[iter 12119] train_loss=0.4855 val_loss=0.4843
Training: 100% 12119/12119 [16:25<00:00, 12.29it/s, loss=0.4845, lr=1.00e-05]
Saved final checkpoint: /content/music-scaling-laws/checkpoints/lstm/lstm_medium_final.pt
Removed resume checkpoint: /content/music-scaling-laws/checkpoints/lstm/lstm_medium_last.pt

Results:
  model_type: lstm
  model_size: medium
  num_params: 23194721
  train_loss: 0.4855
  val_loss: 0.4825
  train_time_sec: 985.9604
  tokens_per_sec: 402770.1207
  gpu_memory_mb: 36183.0698

############################################################
# LSTM - LARGE
############################################################

============================================================
Training lstm (large)
============================================================
Vocab size: 97
Model parameters: 49,319,777
Device: cuda
Model device: cuda:0
Batch size: 128
Block size: 256
Total tokens: 397,146,516
Token budget: 397,146,516
Tokens per iter: 32,768
Max iterations: 12,119 (397.1M tokens)
Training:   4% 500/12119 [01:09<26:37,  7.27it/s, loss=1.0227, lr=2.99e-04]
[iter 500] train_loss=1.0051 val_loss=0.9503
Training:   8% 1000/12119 [02:21<25:24,  7.29it/s, loss=0.7770, lr=2.96e-04]
[iter 1000] train_loss=0.7615 val_loss=0.7382
Training:  12% 1500/12119 [03:33<24:20,  7.27it/s, loss=0.6766, lr=2.91e-04]
[iter 1500] train_loss=0.6670 val_loss=0.6578
Training:  17% 2000/12119 [04:45<23:09,  7.28it/s, loss=0.6185, lr=2.84e-04]
[iter 2000] train_loss=0.6119 val_loss=0.5964
Training:  21% 2500/12119 [05:59<22:04,  7.26it/s, loss=0.5764, lr=2.74e-04]
[iter 2500] train_loss=0.5749 val_loss=0.5626
Training:  25% 3000/12119 [07:11<20:52,  7.28it/s, loss=0.5524, lr=2.63e-04]
[iter 3000] train_loss=0.5483 val_loss=0.5380
Training:  29% 3500/12119 [08:23<19:44,  7.28it/s, loss=0.5344, lr=2.50e-04]
[iter 3500] train_loss=0.5296 val_loss=0.5237
Training:  33% 4000/12119 [09:35<18:36,  7.27it/s, loss=0.5204, lr=2.31e-04]
[iter 4000] train_loss=0.5197 val_loss=0.5121
Training:  37% 4500/12119 [10:48<17:23,  7.30it/s, loss=0.5112, lr=2.15e-04]
[iter 4500] train_loss=0.5104 val_loss=0.5030
Training:  41% 5000/12119 [12:00<16:15,  7.30it/s, loss=0.4994, lr=1.98e-04]
[iter 5000] train_loss=0.5007 val_loss=0.4952
Training:  45% 5500/12119 [13:13<15:08,  7.28it/s, loss=0.4953, lr=1.80e-04]
[iter 5500] train_loss=0.4936 val_loss=0.4922
Training:  50% 6000/12119 [14:25<13:58,  7.30it/s, loss=0.4876, lr=1.62e-04]
[iter 6000] train_loss=0.4886 val_loss=0.4852
Training:  54% 6500/12119 [15:38<12:52,  7.27it/s, loss=0.4833, lr=1.44e-04]
[iter 6500] train_loss=0.4860 val_loss=0.4776
Training:  58% 7000/12119 [16:50<11:42,  7.29it/s, loss=0.4824, lr=1.25e-04]
[iter 7000] train_loss=0.4791 val_loss=0.4751
Training:  62% 7500/12119 [18:03<10:33,  7.29it/s, loss=0.4729, lr=1.08e-04]
[iter 7500] train_loss=0.4749 val_loss=0.4785
Training:  66% 8000/12119 [19:16<09:25,  7.29it/s, loss=0.4695, lr=8.67e-05]
[iter 8000] train_loss=0.4710 val_loss=0.4735
Training:  70% 8500/12119 [20:29<08:15,  7.30it/s, loss=0.4725, lr=7.11e-05]
[iter 8500] train_loss=0.4692 val_loss=0.4708
Training:  74% 9000/12119 [21:41<07:08,  7.29it/s, loss=0.4648, lr=5.69e-05]
[iter 9000] train_loss=0.4636 val_loss=0.4677
Training:  78% 9500/12119 [22:53<06:01,  7.25it/s, loss=0.4669, lr=4.42e-05]
[iter 9500] train_loss=0.4649 val_loss=0.4664
Training:  83% 10000/12119 [24:05<04:51,  7.28it/s, loss=0.4642, lr=3.33e-05]
[iter 10000] train_loss=0.4629 val_loss=0.4642
Training:  87% 10500/12119 [25:19<03:42,  7.29it/s, loss=0.4634, lr=2.43e-05]
[iter 10500] train_loss=0.4640 val_loss=0.4610
Training:  91% 11000/12119 [26:31<02:33,  7.29it/s, loss=0.4617, lr=1.74e-05]
[iter 11000] train_loss=0.4571 val_loss=0.4649
Training:  95% 11500/12119 [27:43<01:25,  7.27it/s, loss=0.4616, lr=1.19e-05]
[iter 11500] train_loss=0.4604 val_loss=0.4676
Training:  99% 12000/12119 [28:56<00:16,  7.26it/s, loss=0.4624, lr=1.01e-05]
[iter 12000] train_loss=0.4621 val_loss=0.4635
Training: 100% 12119/12119 [29:15<00:00,  7.29it/s, loss=0.4610, lr=1.00e-05]
[iter 12119] train_loss=0.4608 val_loss=0.4654
Training: 100% 12119/12119 [29:18<00:00,  6.89it/s, loss=0.4610, lr=1.00e-05]
Saved final checkpoint: /content/music-scaling-laws/checkpoints/lstm/lstm_large_final.pt
Removed resume checkpoint: /content/music-scaling-laws/checkpoints/lstm/lstm_large_last.pt

Results:
  model_type: lstm
  model_size: large
  num_params: 49319777
  train_loss: 0.4608
  val_loss: 0.4622
  train_time_sec: 1758.1287
  tokens_per_sec: 225873.9043
  gpu_memory_mb: 36183.0698

############################################################
# LSTM - XL
############################################################

============================================================
Training lstm (xl)
============================================================
Vocab size: 97
Model parameters: 96,572,769
Device: cuda
Model device: cuda:0
Batch size: 128
Block size: 256
Total tokens: 397,146,516
Token budget: 397,146,516
Tokens per iter: 32,768
Max iterations: 12,119 (397.1M tokens)
Training:   4% 500/12119 [02:02<46:48,  4.14it/s, loss=1.0025, lr=2.99e-04]
[iter 500] train_loss=0.9885 val_loss=0.9445
Training:   8% 1000/12119 [04:10<44:57,  4.12it/s, loss=0.7481, lr=2.96e-04]
[iter 1000] train_loss=0.7357 val_loss=0.7110
Training:  12% 1500/12119 [06:17<42:50,  4.13it/s, loss=0.6488, lr=2.91e-04]
[iter 1500] train_loss=0.6381 val_loss=0.6300
Training:  17% 2000/12119 [08:25<40:55,  4.12it/s, loss=0.5860, lr=2.84e-04]
[iter 2000] train_loss=0.5808 val_loss=0.5709
Training:  21% 2500/12119 [10:34<38:52,  4.12it/s, loss=0.5470, lr=2.74e-04]
[iter 2500] train_loss=0.5426 val_loss=0.5335
Training:  25% 3000/12119 [12:44<36:48,  4.13it/s, loss=0.5250, lr=2.63e-04]
[iter 3000] train_loss=0.5207 val_loss=0.5133
Training:  29% 3500/12119 [14:51<34:52,  4.12it/s, loss=0.5079, lr=2.50e-04]
[iter 3500] train_loss=0.5072 val_loss=0.5015
Training:  33% 4000/12119 [16:59<32:45,  4.13it/s, loss=0.4955, lr=2.31e-04]
[iter 4000] train_loss=0.4950 val_loss=0.4912
Training:  37% 4500/12119 [19:08<30:43,  4.13it/s, loss=0.4865, lr=2.15e-04]
[iter 4500] train_loss=0.4858 val_loss=0.4821
Training:  41% 5000/12119 [21:16<28:47,  4.12it/s, loss=0.4786, lr=1.98e-04]
[iter 5000] train_loss=0.4783 val_loss=0.4821
Training:  45% 5500/12119 [23:23<26:40,  4.13it/s, loss=0.4725, lr=1.80e-04]
[iter 5500] train_loss=0.4717 val_loss=0.4734
Training:  50% 6000/12119 [25:31<24:45,  4.12it/s, loss=0.4682, lr=1.62e-04]
[iter 6000] train_loss=0.4706 val_loss=0.4698
Training:  54% 6500/12119 [27:42<22:39,  4.13it/s, loss=0.4656, lr=1.44e-04]
[iter 6500] train_loss=0.4582 val_loss=0.4649
Training:  58% 7000/12119 [29:49<20:36,  4.14it/s, loss=0.4596, lr=1.25e-04]
[iter 7000] train_loss=0.4572 val_loss=0.4633
Training:  62% 7500/12119 [31:56<18:40,  4.12it/s, loss=0.4555, lr=1.08e-04]
[iter 7500] train_loss=0.4535 val_loss=0.4554
Training:  66% 8000/12119 [34:04<16:37,  4.13it/s, loss=0.4509, lr=8.67e-05]
[iter 8000] train_loss=0.4501 val_loss=0.4549
Training:  70% 8500/12119 [36:13<14:38,  4.12it/s, loss=0.4487, lr=7.11e-05]
[iter 8500] train_loss=0.4490 val_loss=0.4563
Training:  74% 9000/12119 [38:21<12:34,  4.13it/s, loss=0.4442, lr=5.69e-05]
[iter 9000] train_loss=0.4442 val_loss=0.4555
Training:  78% 9500/12119 [40:28<10:34,  4.13it/s, loss=0.4421, lr=4.42e-05]
[iter 9500] train_loss=0.4433 val_loss=0.4497
Training:  83% 10000/12119 [42:36<08:32,  4.13it/s, loss=0.4431, lr=3.33e-05]
[iter 10000] train_loss=0.4424 val_loss=0.4481
Training:  87% 10500/12119 [44:45<06:32,  4.13it/s, loss=0.4426, lr=2.43e-05]
[iter 10500] train_loss=0.4437 val_loss=0.4484
Training:  91% 11000/12119 [46:53<04:31,  4.13it/s, loss=0.4371, lr=1.74e-05]
[iter 11000] train_loss=0.4389 val_loss=0.4477
Training:  95% 11500/12119 [49:01<02:30,  4.12it/s, loss=0.4395, lr=1.19e-05]
[iter 11500] train_loss=0.4394 val_loss=0.4445
Training:  99% 12000/12119 [51:08<00:28,  4.13it/s, loss=0.4390, lr=1.01e-05]
[iter 12000] train_loss=0.4416 val_loss=0.4448
Training: 100% 12119/12119 [51:43<00:00,  4.13it/s, loss=0.4416, lr=1.00e-05]
[iter 12119] train_loss=0.4421 val_loss=0.4430
Training: 100% 12119/12119 [51:47<00:00,  3.90it/s, loss=0.4416, lr=1.00e-05]
Saved final checkpoint: /content/music-scaling-laws/checkpoints/lstm/lstm_xl_final.pt
Removed resume checkpoint: /content/music-scaling-laws/checkpoints/lstm/lstm_xl_last.pt

Results:
  model_type: lstm
  model_size: xl
  num_params: 96572769
  train_loss: 0.4421
  val_loss: 0.4457
  train_time_sec: 3107.6132
  tokens_per_sec: 127787.9067
  gpu_memory_mb: 36183.0698
[train] Results saved to /content/music-scaling-laws/scaling_results.json

Transformer Scaling Law:
  L = 594.3796 * N^(-0.6037) + 0.4010
  Exponent α = 0.6037

LSTM Scaling Law:
  L = 12.1997 * N^(-0.2669) + 0.3540
  Exponent α = 0.2669

Plot saved to: /content/music-scaling-laws/report/scaling_laws.png
Training curves saved to: /content/music-scaling-laws/report/training_curves.png

============================================================
RESULTS TABLE (Markdown)
============================================================
| Model | Size | Parameters | Val Loss | Train Time (s) | Tokens/sec |
|-------|------|------------|----------|----------------|------------|
| lstm | tiny | 925,921 | 0.6650 | 198.1 | 2004873 |
| lstm | small | 3,728,993 | 0.5714 | 351.7 | 1129075 |
| lstm | medium | 23,194,721 | 0.4825 | 986.0 | 402770 |
| lstm | large | 49,319,777 | 0.4622 | 1758.1 | 225874 |
| lstm | xl | 96,572,769 | 0.4457 | 3107.6 | 127788 |
| transformer | tiny | 801,152 | 0.5633 | 490.7 | 809305 |
| transformer | small | 4,176,720 | 0.4595 | 1417.4 | 280173 |
| transformer | medium | 18,643,240 | 0.4280 | 4153.3 | 95615 |
| transformer | large | 55,444,740 | 0.4132 | 10418.1 | 38118 |
| transformer | xl | 99,209,472 | 0.4088 | 16951.7 | 23426 |
Using device: cuda

Evaluating: /content/music-scaling-laws/checkpoints/lstm/lstm_large_final.pt
  Test PPL: 1.57

Evaluating: /content/music-scaling-laws/checkpoints/lstm/lstm_medium_final.pt
  Test PPL: 1.63

Evaluating: /content/music-scaling-laws/checkpoints/lstm/lstm_small_final.pt
  Test PPL: 1.77

Evaluating: /content/music-scaling-laws/checkpoints/lstm/lstm_tiny_final.pt
  Test PPL: 1.95

Evaluating: /content/music-scaling-laws/checkpoints/lstm/lstm_xl_final.pt
  Test PPL: 1.55

Evaluating: /content/music-scaling-laws/checkpoints/transformer/transformer_large_final.pt
  Test PPL: 1.52

Evaluating: /content/music-scaling-laws/checkpoints/transformer/transformer_medium_final.pt
  Test PPL: 1.52

Evaluating: /content/music-scaling-laws/checkpoints/transformer/transformer_small_final.pt
  Test PPL: 1.58

Evaluating: /content/music-scaling-laws/checkpoints/transformer/transformer_tiny_final.pt
  Test PPL: 1.74

Evaluating: /content/music-scaling-laws/checkpoints/transformer/transformer_xl_final.pt
  Test PPL: 1.50

============================================================
EVALUATION SUMMARY
============================================================
Model              Params    Val PPL   Test PPL
------------------------------------------------------------
transformer       801,152       1.75       1.74
lstm              925,921       1.96       1.95
lstm            3,728,993       1.76       1.77
transformer     4,176,720       1.59       1.58
transformer    18,643,240       1.54       1.52
lstm           23,194,721       1.62       1.63
lstm           49,319,777       1.58       1.57
transformer    55,444,740       1.51       1.52
lstm           96,572,769       1.57       1.55
transformer    99,209,472       1.50       1.50
[evaluate] Plots saved to /content/music-scaling-laws/report
[generate] Using checkpoint: /content/music-scaling-laws/checkpoints/transformer/transformer_xl_final.pt
Using device: cuda
Model: TransformerLM
Parameters: 99,209,472
Vocab size: 97

--- Sample 1 (valid=True) ---
X:1
T:Untitled
M: 4/4
L: 1/8
Q:1/4=120
K:C % 0 sharps
z8| z8| z8| z8|
z8| z8| z8| z8|
z8| z8| z8| z8|
z8| z8| z8| z8|
z8| z8| z8| z8|
z8| z8| z8| z8|
z8| z8| z8| z8|
z8| z8| z8| z8|
z8| z8| z8| z8|
z8...

--- Sample 2 (valid=True) ---
X:1
T:Untitled
M: 4/4
L: 1/8
Q:1/4=120
K:G % 1 sharps
z8| z8| z8| z8|
z8| z8| z8| z8|
z8| z8| z8| z8|
z8| z8| z8| z8|
z8| z8| z8| z8|
z8| z8| z8| z8|
z8| z8| z8| z8|
z8| z8| z8| z8|
z8| z8| z8| z8|
z8...

--- Sample 3 (valid=True) ---
X:1
T:Untitled
M: 3/4
L: 1/8
Q:1/4=81
K:G % 1 sharps
z4  z F| [B- B,,,] B/2- [B- D B, F,]/2  B/2- [B D B, F,]/2 z  [B D B, F,] z| [G- B,,,] G/2- [G- D B, F,]/2  G/2- [G D B, F,]/2 z  [G- D B, F,] G/2 ...

--- Sample 4 (valid=True) ---
X:1
T:Untitled
M: 4/4
L: 1/8
Q:1/4=139
K:G % 1 sharps
z8| z8| z8| z8|
z8| z8| z4  z3/2
[D- ^A,- =F,- G,,-]2 [D- A,- F,- G,,-]/2| [D- ^A,- =F, G,,-] [D- A,- G,,]/2 [D- A,- F,-] [D- A,- F,- D,-]3 [D- A,...

--- Sample 5 (valid=True) ---
X:1
T:Untitled
M: 4/4
L: 1/8
Q:1/4=120
K:A % 3 sharps
z8| A,,,2  z A,,,  E,,2  A,,,2| E,,2-  E,,/2 z/2 B,,,2 E,,2 F,,-| F,,2  F,, F,,  B,,,2  B,,,2|
E,,2  z E,,  B,,,2  E,,2| A,,,2  z A,,,  E,,2  A,,,...

--- Sample 6 (valid=True) ---
X:1
T:Untitled
M: 4/4
L: 1/8
Q:1/4=65
K:F % 1 flats
z8| z8| z8| z8|
z2  z/2 c''3/2  d''3- d''/2 c''/2-| c''3/2 f'6- f'/2-| f'4-  f' a'  c'' f'| c'' f'  f'3/2 c'' c'' c''/2  c'' c''|
z/2 c'' c'' z/2 d'...

--- Sample 7 (valid=True) ---
X:1
T:Untitled
M: 4/8
L: 1/16
Q:1/4=80
K:C % 0 sharps
[g e]2  [f ^d]2  [e- c-]3 [e c]/2 z/2| [d B]2  [c A]2  z3 [B ^G]| [c A]2  [B ^G]2  [A- F-]3 [A F]/2 z/2| [c A]2  [B ^G]2  [A- F]3/2 A/2  [B G]2|
[...

--- Sample 8 (valid=True) ---
X:1
T:Untitled
M: 4/4
L: 1/8
Q:1/4=160
K:Eb % 3 flats
z8| z8| z8| z8|
z8| z8| z8| b8-|
b8| b8| b8| b8|
b8| b8| b8| b8|
b8| b8| b8| b8|
b8| b8| b8| b8|
b8| b8| b8| b8|
b8| b8| b8| b8|
b8| b8| b8| b8|
b...

--- Sample 9 (valid=True) ---
X:1
T:Untitled
M: 4/4
L: 1/8
Q:1/4=100
K:C % 0 sharps
z8| z8| z8| z8|
z8| z8| z8| z8|
z8| z A,,/2 z/2  A,, z/2 A,,/2  z/2 A,, z/2  A,, z/2 A,,/2| z/2 A,, z/2  A,, z/2 A,,/2  z/2 A,, z/2  A,, z/2 A,,/2...

--- Sample 10 (valid=True) ---
X:1
T:Untitled
M: 3/4
L: 1/8
Q:1/4=120
K:C % 0 sharps
z6| z6| z6| z6|
z6| z6| z6| z6|
z6| z6| z6| z6|
z6| z6| z6| z6|
z6| z6| z6| z6|
z6| z2
^A,3/2 z/2  ^G,3/2 z/2| ^A,3/2 z/2  ^G,3/2 z/2  A,3/2 z/2| ...
MIDI conversion failed: Bad chord indicator: [: no closing bracket found.
MIDI conversion failed: Bad chord indicator: [e: no closing bracket found.
MIDI conversion failed: Bad chord indicator: [D- A,- F: no closing bracket found.

============================================================
GENERATION SUMMARY
============================================================
Total samples: 10
Valid ABC: 10 (100.0%)
MIDI conversions: 7/10 (70.0%)
Output directory: /content/music-scaling-laws/samples
[generate] Samples saved to /content/music-scaling-laws/samples
Pipeline complete.