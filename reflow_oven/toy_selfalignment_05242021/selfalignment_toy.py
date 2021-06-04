import numpy as np
import matplotlib.pyplot as plt

class chipSize(object):
    def __init__(self, chip):
        if chip == 'R0402' :
            self.l = 400
            self.w = 200
            self.t = 50/8 # tension level; 6.25
        elif chip == 'R0603' :
            self.l = 600
            self.w = 300
            self.t = 50/18 # 2.78
        elif chip == 'R1005' :
            self.l = 1000
            self.w = 500
            self.t = 1     # 1

def linearShift(pre_data, direction = 0, d_noise = 10, distance = 40, noise = 20) :
    # pre_data : n x d([pre_x, pre_y])
    # direction : angle(degree)
    # d_noise : angle
    # distance
    # noise
    outs = []
    for datum in pre_data :
        # d = [pre_x, pre_y]
        _rad = np.radians(np.random.normal(loc = direction, scale = d_noise))
        dx = np.random.normal(loc = distance, scale = noise) * np.cos(_rad)
        dy = np.random.normal(loc = distance, scale = noise) * np.sin(_rad)
        outs.append([datum[0] + dx, datum[1] + dy])
    return np.asarray(outs)

def tensionSimple(pre_data, chip = 'R0402', alpha = 0.1):
    assert chip in ['R0402','R0603','R1005']
    ch = chipSize(chip)
    outs = []
    for datum in pre_data :
        pre_x, pre_y, SPI_x, SPI_y, SPI_diff = datum

        rel_SPI_x = (SPI_x - pre_x) * (1 - SPI_diff)
        rel_SPI_y = SPI_y - pre_y
            
        post_x = pre_x + rel_SPI_x * ch.t * alpha
        post_y = pre_y + rel_SPI_y * ch.t * alpha
        outs.append([post_x, post_y])
    return np.asarray(outs)

def tension(pre_data, chip = 'R0402', alpha = 0.1, beta = 0.0005) :
    assert chip in ['R0402','R0603','R1005']
    ch = chipSize(chip)
    outs = []
    # pre_data : n x d([pre_x, pre_y, pre_theta, SPI_x1, SPI_y1, SPI_x2, SPI_y2, SPI_vol1, SPI_vol2])
    # pre_theta : clockwise
    # SPI_vol([0,1])
    for datum in pre_data :
        pre_x, pre_y, pre_theta, SPI_x1, SPI_y1, SPI_x2, SPI_y2, SPI_vol1, SPI_vol2 = datum
        rad = np.radians(pre_theta)
        _rad = -rad
        rev_rot_mat = np.array([[np.cos(_rad), -np.sin(_rad), 0],
                            [np.sin(_rad), np.cos(_rad), 0],
                            [0, 0, 1]])

        rel_SPI_x1 = SPI_x1 - pre_x
        rel_SPI_x2 = SPI_x2 - pre_x
        rel_SPI_y1 = SPI_y1 - pre_y
        rel_SPI_y2 = SPI_y2 - pre_y

        spi_coord = np.array([[rel_SPI_x1, rel_SPI_x2],
                                [rel_SPI_y1, rel_SPI_y2],
                                [1,1]])
        rev_rot_spi = np.matmul(rev_rot_mat, spi_coord)

        rev_rot_spi += np.array([[-ch.l/2, ch.l/2],
                                 [-ch.w/2, ch.w/2],
                                 [0,0]])
        rev_rot_spi = np.matmul(rev_rot_spi, np.array([[SPI_vol1, SPI_vol1],
                                                       [SPI_vol2, SPI_vol2]]))
        tension = np.sum(rev_rot_spi, 0)[:2] * ch.t # [transpose, rotation]

        post_x = pre_x + alpha * tension[0] * np.cos(rad)
        post_y = pre_y + alpha * tension[0] * np.sin(rad)
        post_theta = pre_theta + np.rad2deg(beta * tension[1])
        outs.append([post_x, post_y, post_theta])
    return np.asarray(outs)


def generatePoints2D(num, mu = 0, sigma = 20) :
    x = np.random.normal(mu, sigma, num).reshape(-1,1)
    y = np.random.normal(mu, sigma, num).reshape(-1,1)
    outs = np.concatenate([x, y], axis = 1)
    return outs

def generateSPIsimple(pre_data, chip = 'R0402') :
    outs = []
    ch = chipSize(chip)
    for datum in pre_data :
        pre_x, pre_y = datum
        SPI_x = pre_x + np.random.uniform(low = 0, high = ch.l * 0.3)
        SPI_y = pre_y + np.random.uniform(low = 0, high = ch.w * 0.3)
        vol_diff = np.random.uniform(low = 0, high = 0.3)
        outs.append([pre_x, pre_y, SPI_x, SPI_y, vol_diff])
    return np.asarray(outs)

def generateSPIinfo(pre_data, chip = 'R0402') :
    outs = []
    ch = chipSize(chip)
    for datum in pre_data :
        pre_x, pre_y = datum
        pre_theta = np.random.uniform(low = 0, high=15) # [degree]
        sides = np.array([[-ch.l/2 + np.random.normal(0, ch.l * 0.05), -ch.w/2 + np.random.normal(0, ch.w * 0.05)],
                          [ch.l/2 + np.random.normal(0, ch.l * 0.05), ch.w/2 + np.random.normal(0, ch.w * 0.05)]]) # [left, right]
        SPI_x1 = sides[0][0] + pre_x
        SPI_y1 = sides[0][1] + pre_y
        SPI_x2 = sides[1][0] + pre_x
        SPI_y2 = sides[1][1] + pre_y

        vols = [np.random.uniform(low = 0.7, high=1.0), np.random.uniform(low = 0.7, high=1.0)]
        SPI_vol1, SPI_vol2 = vols
        outs.append([pre_x, pre_y, pre_theta, SPI_x1, SPI_y1, SPI_x2, SPI_y2, SPI_vol1, SPI_vol2])
    return np.asarray(outs)

if __name__ == '__main__':
    train_sample = 100

    inputs = generatePoints2D(train_sample, mu = 20, sigma = 40)
    posts = linearShift(inputs)

    fig, ax = plt.subplots()
    ax.scatter(inputs[:,0], inputs[:,1], label = 'inputs')
    ax.scatter(posts[:,0], posts[:,1], label ='outputs')
    ax.legend(fontsize=12, loc='upper left')  # legend position
    plt.ylim([-150, 150])
    plt.xlim([-150, 150])
    plt.title('generatePoints2D, linearShift')
    # plt.savefig('fig1.png', dpi=300)


    exp_inputs = generateSPIsimple(inputs)
    exp_posts = tensionSimple(exp_inputs)

    fig, ax = plt.subplots()
    ax.scatter(exp_inputs[:,0], exp_inputs[:,1], label = 'inputs')
    ax.scatter(exp_posts[:,0], exp_posts[:,1], label ='outputs')
    ax.legend(fontsize=12, loc='upper left')  # legend position
    plt.ylim([-150, 150])
    plt.xlim([-150, 150])
    plt.title('generateSPIsimple, tensionSimple')
    # plt.savefig('fig2.png', dpi=300)

    full_inputs = generateSPIinfo(inputs)
    full_posts = tension(full_inputs)

    fig, ax = plt.subplots()
    ax.scatter(full_inputs[:,0], full_inputs[:,1], label = 'inputs')
    ax.scatter(full_posts[:,0], full_posts[:,1], label = 'outputs')
    ax.legend(fontsize=12, loc='upper left')
    plt.ylim([-150,150])
    plt.xlim([-150,150])
    plt.title('generateSPIinfo, tension')
    # plt.savefig('fig3.png', dpi=200)

    plt.show()