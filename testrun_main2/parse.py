import matplotlib.pyplot as plt

with open('text.txt', 'r') as file:
    lines = file.readlines()
    pres, posts = [], []
    for line in lines:
        tokens = line.split('->')
        pre = float(tokens[0][-7:])
        post = float(tokens[1][1:8])
        pres.append(pre)
        posts.append(post)
    # with open('pres.txt','w') as writefile:
    #     for item in pres:
    #         writefile.write(str(item))
    #         writefile.write('\n')
    # with open('posts.txt','w') as writefile:
    #     for item in posts:
    #         writefile.write(str(item))
    #         writefile.write('\n')

    # fig = plt.figure()
    # ax = fig.add_subplot()
    # ax.bar(pres,posts)
    # # ax.set_xlabel(pres)
    # ax.set_xlabel('Pre-AOI offset')
    # ax.set_ylabel('Post-AOI offset')
    # ax.set_title('Pre vs. Post')
    # fig.savefig('distances.png')
    