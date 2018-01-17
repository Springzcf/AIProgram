# import random
# an1 = 1
# an2 = 2
# for i in range(1,10):
#     an = an1*an2
#     an1 = an2
#     an2 = an
#     print(an,end='\t')
#     # for j in range(10):
#     #     print(i**(3)+2,end='\t')
#     if i%4 ==0:
#         print()
# 立方+1
def q1():
    for i in range(1,1000):
        print(i**2,end='\t')
        if i%4 ==0:
            print()

# 产生质数
def zs():
    i = 1
    p_num=3
    listZS=[]
    while p_num < 40000:
        div_num = 2
        while div_num < p_num:
            if p_num % div_num == 0:
                break
            else:
                div_num += 1
        if div_num == p_num :
            listZS.append(p_num)
        p_num += 1
    for zs in listZS:
        print(zs,end='\t')
        if i%4 == 0:
            print()
        i+=1

# 第三项为第二项*2+第一项


# a = i*(i-1)
def qt():
    for i in range(10000):
        print(i*(i-1),end='\t')
        if i%4 == 0:
            print()
qt()