x = int(input("please enter a number... "))


def f(x):
  i = 0
  u = 0
  t = 0

  while True:
    if x%2==0:
        x=x/2
        u+=1
    else:
        x=3*x+1
        t+=1
    i+=1

    if x==4:
      return i,u,t
  

for i in range(1,x+1):
   print(f(i))