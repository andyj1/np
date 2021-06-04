import numpy as np
import matplotlib.pyplot as plt

class chipSize(object):
    '''
    l, w: length, width of the chip
    t: resistance in movement due to chip specifications (increases with chip size)
    '''
    def __init__(self, chip):
        if chip == 'R0402' :
            self.l = 400
            self.w = 200
            self.t = 50/8       # tension level: 6.25
        elif chip == 'R0603' :
            self.l = 600
            self.w = 300
            self.t = 50/18      # tension level: 2.78
        elif chip == 'R1005' :
            self.l = 1000
            self.w = 500
            self.t = 1          # tension level: 1.00

def linearShift(pre_data, direction = 0, d_noise = 10, distance = 40, noise = 20):
    '''
    constant shift variables:
        angle (deg): randn(mean=0, std=10)
        position: randn(mean=40, std=20)
        
        total shift (dx, dy) = position * angle
        
    return
        original (x,y) + shift (dx, dy)
    
    '''
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
    '''
    simple self-alignment effect
        :demonstrates chip movement towards solder paste in the reflow oven
        :adds solder paste volume difference aspect (by multiplying to relative position difference in x)
        :adds resistance aspect for each chip
        :adds alpha for some additional variables not accounted for
    
    return
        (x,y) + (solder paste x,y - chip x,y) * (1-solder paste volume difference)
        (= estimated chip position inspected at Post-AOI stage)
        
    '''
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

def tension(pre_data, chip = 'R0402', alpha = 0.5, beta = 0.0005) :
    '''
    1. apply 2D rotation (ccw or cw) and translation
    2. apply some other operation using volumes at pads 1 and 2 ...?
        sum of (2) result <-- "tension"
    3. multiply this tension by x,y component of rotated angle again ...? 
        add this to pre-aoi chip (x,y)
    
    '''
    assert chip in ['R0402','R0603','R1005']
    ch = chipSize(chip)
    outs = []
    # pre_data : n x d([pre_x, pre_y, pre_theta, SPI_x1, SPI_y1, SPI_x2, SPI_y2, SPI_vol1, SPI_vol2])
    # pre_theta : clockwise
    # SPI_vol([0,1])
    for datum in pre_data :
        pre_x, pre_y, pre_theta, SPI_x1, SPI_y1, SPI_x2, SPI_y2, SPI_vol1, SPI_vol2 = datum
        rad = np.radians(pre_theta) # rotate by pre theta...?
        _rad = -rad
        # ccw rotation about the origin in 2D for _rad (in this case is negative, so ultimately clockwise)
        # x' = xcos(theta) - ysin(theta)
        # y' = xsine(theta) + ycos(theta)


        rel_SPI_x1 = SPI_x1 - pre_x
        rel_SPI_x2 = SPI_x2 - pre_x
        rel_SPI_y1 = SPI_y1 - pre_y
        rel_SPI_y2 = SPI_y2 - pre_y
        spi_coord = np.array([[rel_SPI_x1, rel_SPI_x2],
                                [rel_SPI_y1, rel_SPI_y2],
                                [1,1]])
        # 2D rotation
        rev_rot_mat = np.array([[np.cos(_rad), -np.sin(_rad), 0],
                                [np.sin(_rad), np.cos(_rad), 0],
                                [0, 0, 1]])
        rev_rot_spi = np.matmul(rev_rot_mat, spi_coord) # 3x3 * 3x2 = 3x2

        # 2D translation ... subtract?
        rev_rot_spi -= np.array([[-ch.l/2, ch.l/2],
                                 [-ch.w/2, ch.w/2],
                                 [0,0]]) # 3x2
        
        # operation using volumes?? 
        # transformed (relative) spi x,y and volumes at pad 1, 2 relationship?
        rev_rot_spi = np.matmul(rev_rot_spi, np.array([[-SPI_vol1, SPI_vol1],
                                                       [-SPI_vol2, SPI_vol2]]))  
                                                       # 3x2 * 2x2 = 3x2
        tension = np.sum(rev_rot_spi, axis=0)[:2] * ch.t # [transpose, rotation]
        # 1x2 * tension level per chip specification

        post_x = pre_x + ( alpha * tension[0] * np.cos(rad) )
        post_y = pre_y + ( alpha * tension[0] * np.sin(rad) )
        post_theta = pre_theta + np.rad2deg(beta * tension[1])
        outs.append([post_x, post_y, post_theta])
    return np.asarray(outs)


def generatePoints2D(num, mu = 0, sigma = 20) :
    x = np.random.normal(mu, sigma, num).reshape(-1,1)
    y = np.random.normal(mu, sigma, num).reshape(-1,1)
    outs = np.concatenate([x, y], axis = 1)
    return outs

def generateSPIsimple(pre_data, chip = 'R0402') :
    # pre data: generatePoints2D
    outs = []
    ch = chipSize(chip)
    for datum in pre_data :
        pre_x, pre_y = datum
        
        SPI_x = pre_x + np.random.uniform(low = 0, high = ch.l * 0.2) - ch.l * 0.1 # added
        SPI_y = pre_y + np.random.uniform(low = 0, high = ch.w * 0.2) - ch.w * 0.1 # added 
        vol_diff = np.random.uniform(low = 0, high = 0.3)
        outs.append([pre_x, pre_y, SPI_x, SPI_y, vol_diff])
    return np.asarray(outs)

def generateSPIinfo(pre_data, chip = 'R0402') :
    # pre data: generatePoints2D
    outs = []
    ch = chipSize(chip)
    for datum in pre_data :
        pre_x, pre_y = datum
        pre_theta = np.random.uniform(low = 0, high=30) - 15 # [degree]
        sides = np.array([[-ch.l/2 + np.random.uniform(0, ch.l * 0.1) - ch.l * 0.1/2, np.random.uniform(0, ch.w * 0.1) - ch.w * 0.1/2], # added
                          [ch.l/2 + np.random.uniform(0, ch.l * 0.1) - ch.l * 0.1/2, np.random.uniform(0, ch.w * 0.1) - ch.w * 0.1/2]]) # [left, right] # added
        SPI_x1 = sides[0][0] + pre_x
        SPI_y1 = sides[0][1] + pre_y
        SPI_x2 = sides[1][0] + pre_x
        SPI_y2 = sides[1][1] + pre_y

        vols = [np.random.uniform(low = 0.7, high=1.0), np.random.uniform(low = 0.7, high=1.0)]
        SPI_vol1, SPI_vol2 = vols
        outs.append([pre_x, pre_y, pre_theta, SPI_x1, SPI_y1, SPI_x2, SPI_y2, SPI_vol1, SPI_vol2])
    return np.asarray(outs)

if __name__ == '__main__':
    train_sample = 1000

    inputs = generatePoints2D(train_sample, mu = 20, sigma = 40)
    posts = linearShift(inputs)

    fig, ax = plt.subplots()
    ax.scatter(inputs[:,0], inputs[:,1], label = 'inputs')
    ax.scatter(posts[:,0], posts[:,1], label ='outputs')
    ax.legend(fontsize=12, loc='upper left')  # legend position
    plt.ylim([-150, 150])
    plt.xlim([-150, 150])
    plt.title('generatePoints2D, linearShift')
    plt.savefig('fig1.png', dpi=300)


    exp_inputs = generateSPIsimple(inputs)
    exp_posts = tensionSimple(exp_inputs)

    fig, ax = plt.subplots()
    ax.scatter(exp_inputs[:,0], exp_inputs[:,1], label = 'inputs')
    ax.scatter(exp_posts[:,0], exp_posts[:,1], label ='outputs')
    ax.legend(fontsize=12, loc='upper left')  # legend position
    plt.ylim([-150, 150])
    plt.xlim([-150, 150])
    plt.title('generateSPIsimple, tensionSimple')
    plt.savefig('fig2.png', dpi=300)

    full_inputs = generateSPIinfo(inputs)
    full_posts = tension(full_inputs)

    fig, ax = plt.subplots()
    ax.scatter(full_inputs[:,0], full_inputs[:,1], label = 'inputs')
    ax.scatter(full_posts[:,0], full_posts[:,1], label = 'outputs')
    ax.legend(fontsize=12, loc='upper left')
    plt.ylim([-150,150])
    plt.xlim([-150,150])
    plt.title('generateSPIinfo, tension')
    plt.savefig('fig3.png', dpi=200)

    for i in range(10) :
        full_input = full_inputs[i]
        full_post = full_posts[i]

        fig, ax = plt.subplots()
        ax.scatter(full_input[0], full_input[1], label = 'input')
        ax.scatter(full_post[0], full_post[1], label = 'output')
        ax.scatter(full_input[3], full_input[4], label = 'SPI1', s=100*full_input[7]**2)
        ax.scatter(full_input[5], full_input[6], label = 'SPI2', s=100*full_input[8]**2)
        ax.legend(fontsize=12, loc='upper left')
        plt.ylim([-300,300])
        plt.xlim([-300,300])
        plt.savefig(f'SPI_example_{i}.png', dpi=200)