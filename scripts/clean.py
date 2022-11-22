import shutil
import os


def main():
    exp_root = "exp"
    all_exp_dirs = os.listdir(exp_root)
    for idir in all_exp_dirs:
        idir = os.path.join(exp_root, idir)
        files = os.listdir(idir)
        if "state_dict.pt" not in files:
            print(idir)
            shutil.rmtree(idir)
            continue


if __name__ == "__main__":
    main()