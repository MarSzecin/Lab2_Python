import numpy as np

"""SAMOUCZEK"""
"""
print('Hello Word')

tekst = 'Witam, co słychac'
print(tekst)

spalanie = 6.5
trasa = 2.0
cena = 5.0
koszt = spalanie*trasa*cena
print(koszt)
print(koszt/5)

jakas_zmienna = "To jest jakis napis"
print(jakas_zmienna)

lubie_placki = True
print(lubie_placki)

lista = [1,2,3,4,"napis",True,False,8.5]
slownik = {"jablko": "ciasto" ,"czy lubisz jablko" : True}

print(lista,slownik)


d = [10, 8, 10, 12, 6, 8, 14, 9]

print(d)
print(d[4])
d.append(14)
print(d)


for i in range(len(d)):
    d[i] = d[i]*0.1
    
for element in d:
    print(element)
    
for element in [1,2,3,4,5]:
    print(element)
    
zmienna = 8
if zmienna == 1:
    print("Wartosc zmiennej to jeden")
else:
    if zmienna == 2:
        print("Wartosc zmiennej to dwa")
    else:
        print("Wartosc zmiennej nie jest ani jeden ani dwa")
    

x=[1,2,3,4,5]
while x:
    y=x.pop()
    print ("Ostatnia wartości z listy x to ", y)
else:
    print ('Koniec')
    
    
def f(x):
    return (2*(x**3))/8.51

print(f(5))

def potega(x, y):
    return x**y
z=potega(3, 2)
print(z)
"""

""" NUMPY i MATPLOTLIB """
# arr = np.array([1, 2, 3, 4, 5])
# print(arr)

# A = np.array([[1, 2, 3], [7, 8, 9]])
# print(A)

A = np.array([[1, 2, \
                3],
              [7, 8, 9]])
# print(A)

# v = np.arange(1,7)
# print(v,"\n")
# v = np.arange(-2,7)
# print(v,"\n")
# v = np.arange(1,10,3)
# print(v,"\n")
# v = np.arange(1,10.1,3)
# print(v,"\n")
# v = np.arange(1,11,3)
# print(v,"\n")
# v = np.arange(1,2,0.1)
# print(v,"\n")

# v = np.linspace(1,3,4)
# print(v)
# v = np.linspace(1,10,4)
# print(v)
# v = np.linspace(0,10,3)
# print(v)

# X = np.ones((2,3))
# Y = np.zeros((2,3,4))
# Z = np.eye(2) # np.eye(2,2) np.eye(2,3)
# Q = np.random.rand(2,5) # np.round(10*np.random.rand((3,3)))

# print(X,"\n\n",Y,"\n\n",Z,"\n\n",Q)
# U = np.block([[A], [X]])
# print(U)

# V = np.block([[
#     np.block([
#         np.block([[np.linspace(1,3,3)],[
#             np.zeros((2,3))]]) ,
#         np.ones((3,1))])
#     ],
#     [np.array([100, 3, 1/2, 0.333])]] )
# print(V)

# print( V[0,2] )
# print( V[3,0] )
# print( V[3,3] )
# print( V[-1,-1] )
# print( V[-4,-3] )

# print( V[3,:] )
# print( V[:,2] )
# print( V[3,0:3] )
# print( V[np.ix_([0,2,3],[0,-1])] )
# print( V[3] )

# Q = np.delete(V, 3, 0)
# print(Q)
# Q = np.delete(V, 2, 1)
# print(Q)
# v = np.arange(1,7)
# print(v)
# print( np.delete(v, 3, 0) )

# print(np.size(v))
# print(np.shape(v))
# print(np.size(V))
# print(np.shape(V))

A = np.array([[1, 0, 0],
              [2, 3, -1],
              [0, 7, 2]] )

B = np.array([[1, 2, 3],
              [-1, 5, 2],
              [2, 2, 2]] )
# print( A+B )
# print( A-B )
# print( A+2 )
# print( 2*A )


# MM1 = A@B
# print(MM1)
# MM2 = B@A
# print(MM2)

# MT1 = A*B
# print(MT1)
# MT2 = B*A
# print(MT2)

# C = np.linalg.solve(A,MM1)
# print(C) 
# x = np.ones((3,1))
# b =  A@x
# y = np.linalg.solve(A,b)
# print(y)

# PM = np.linalg.matrix_power(A,2) 
# print(PM)
# PT = A**2  
# print(PT)

# transpozycja
print(A.T)
print(A.transpose())
# hermitowskie sprzezenie macierzy (dla m. zespolonych)
print(A.conj().T)
print(A.conj().transpose())










    
