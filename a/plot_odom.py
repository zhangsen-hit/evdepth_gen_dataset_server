import matplotlib.pyplot as plt


def parse_odometry(file_path):

    seqs = []
    px, py, pz = [], [], []
    ox, oy, oz = [], [], []

    with open(file_path, 'r') as f:
        content = f.read()

    # 每个ros message
    blocks = content.split('---')

    for block in blocks:

        lines = block.split('\n')

        seq = None
        pos = {}
        ori = {}

        mode = None

        for line in lines:

            line = line.strip()

            if line.startswith("seq:"):
                seq = int(line.split(":")[1])

            elif line.startswith("position:"):
                mode = "pos"

            elif line.startswith("orientation:"):
                mode = "ori"

            elif line.startswith("covariance"):
                mode = None

            elif line.startswith("x:"):
                if mode == "pos":
                    pos["x"] = float(line.split(":")[1])
                elif mode == "ori":
                    ori["x"] = float(line.split(":")[1])

            elif line.startswith("y:"):
                if mode == "pos":
                    pos["y"] = float(line.split(":")[1])
                elif mode == "ori":
                    ori["y"] = float(line.split(":")[1])

            elif line.startswith("z:"):
                if mode == "pos":
                    pos["z"] = float(line.split(":")[1])
                elif mode == "ori":
                    ori["z"] = float(line.split(":")[1])

        # 只记录完整数据
        if seq is not None and len(pos) == 3 and len(ori) == 3:

            seqs.append(seq)

            px.append(pos["x"])
            py.append(pos["y"])
            pz.append(pos["z"])

            ox.append(ori["x"])
            oy.append(ori["y"])
            oz.append(ori["z"])

    return seqs, px, py, pz, ox, oy, oz


def plot_odometry(file_path):

    seqs, px, py, pz, ox, oy, oz = parse_odometry(file_path)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    axes[0,0].plot(seqs, px)
    axes[0,0].set_title("Position X")

    axes[0,1].plot(seqs, py)
    axes[0,1].set_title("Position Y")

    axes[0,2].plot(seqs, pz)
    axes[0,2].set_title("Position Z")

    axes[1,0].plot(seqs, ox)
    axes[1,0].set_title("Orientation X")

    axes[1,1].plot(seqs, oy)
    axes[1,1].set_title("Orientation Y")

    axes[1,2].plot(seqs, oz)
    axes[1,2].set_title("Orientation Z")

    for ax in axes.flat:
        ax.set_xlabel("seq")
        ax.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    plot_odometry("odometry.txt")
