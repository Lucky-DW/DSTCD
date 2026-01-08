def img_cut(img_x, img_y, cut_x, cut_y, i):
    x = img_x // cut_x
    y = img_y // cut_y
    if i / x <= 1 :
        x_point = (i - 1) * cut_x
        y_point = 0
    elif i % x == 0:
        x_point = (x-1)*cut_x
        y_point = ((i-1) // x) * cut_y
    else:
        x_point = ((i % x)-1) * cut_x
        y_point = ((i-1) // x) * cut_y
    return x_point, y_point


if __name__ == '__main__':
    img_x = 5000
    img_y = 5000
    cut_x =1000
    cut_y =1000
    cut_num = (img_x//cut_x)*(img_y//cut_y)
    for i in range(cut_num):
        x,y = img_cut(img_x, img_y, cut_x, cut_y, i+1)
        print("第",i+1,"块")
        print(x,y)
