import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--is_train', type=bool, default=True)
    # parser.add_argument('--checkpoint', type=str, default=None)


    parser.add_argument('--is_train', type=bool, default=True)
    parser.add_argument('--checkpoint', type=str, default='checkpoint_3600.pth')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--data_type', type=str, default='rgb')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--w', type=int, default=112)
    parser.add_argument('--h', type=int, default=112)
    parser.add_argument('--n_frames_per_clip', type=int, default=16)
    args = parser.parse_args()
    return args

# def parse_opts():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--is_train', type=bool, default=True)
#     parser.add_argument('--checkpoint', type=str, default='checkpoint_3600.pth')
#     parser.add_argument('--num_workers', type=int, default=8)
#     parser.add_argument('--default_device', type=int, default=1)
#     parser.add_argument('--batch_size_1', type=int, default=256)
#     parser.add_argument('--batch_size_2', type=int, default=1)
#     parser.add_argument('--batch_size_3', type=int, default=256)
#     parser.add_argument('--w', type=int, default=224)
#     parser.add_argument('--h', type=int, default=126)
#     parser.add_argument('--n_frames_per_clip', type=int, default=100)
#     parser.add_argument('--lr_patience', type=int, default=1)
#     args = parser.parse_args()
#     return args