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
"""

d = [10, 8, 10, 12, 6, 8, 14, 9]
"""
print(d)
print(d[4])
d.append(14)
print(d)
"""

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

    
