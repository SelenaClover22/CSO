import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description="MMBD-CSO")

    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to a model state_dict (.pth/.pt/.pt.tar). "
                             "Supports raw state_dicts and dicts wrapping the weights under "
                             "'netC' / 'model' / 'net'.")
    parser.add_argument("--save_dir", type=str, default="./results",
                        help="Directory for results.npy / results_mean.npy.")
    parser.add_argument("--data_root", type=str, default = "./data")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--size", type=int, default=32, help="Model input spatial size.")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "preactresnet18"])

    parser.add_argument("--num_clean_samples", type=int, default=10,
                        help="Clean images per class used for mask + margin optimization.")
    parser.add_argument("--mround", type=int, default=20,
                        help="Steps of mask-decoupling optimization per class.")
    parser.add_argument("--margin_steps", type=int, default=500,
                        help="Margin-maximization steps per class.")
    parser.add_argument("--cos_sim_ratio", type=float, default=400.0,
                        help="Weight on the cosine-similarity loss during margin maximization.")
    parser.add_argument("--main_lr", type=float, default=1e-3,
                        help="Base learning rate; boosted 10x once cos_sim drops below ~1e-2.")

    return parser
