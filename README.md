# Laba1
Лабораторная 1 для АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ

Отчет по лабораторной работе #1 выполнил(а):
- Голубятникова Ксения Александровна
- РИ210939
Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | * | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Структура отчета

- Данные о работе: название работы, фио, группа, выполненные задания.
- Цель работы.
- Задание 1.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 2.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 3.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Выводы.
- ✨Magic ✨

## Цель работы
Ознакомиться с основными операторами зыка Python на примере реализации линейной регрессии.

## Задание 1
### Написание программ Hello World на Python и Unity
- Для Python с демонстрацией сохранения документа goojle.collab
![image](https://user-images.githubusercontent.com/114469025/192469514-e13691c9-db4d-4001-b9de-499163f7383d.png)
![image](https://user-images.githubusercontent.com/114469025/192469552-48bfe473-9034-4470-85d1-9d2635dff3f5.png)
- Для Unity вывод в консоль


## Задание 2
### Пошагово выполнить каждый пункт раздела "ход работы" с описанием и примерами реализации задач
Ход работы:
1. Произвести подготовку данных для работы с алгоритмом линейной регрессии. 10 видов данных были установлены случайным образом, и данные находились в линейной зависимости. Данные преобразуются в формат массива, чтобы их можно было вычислить напрямую при использовании умножения и сложения.

```py

#Import the required modules, numpy for calculation, and Matplotlib for drawing
import numpy as np
import matplotlib.pyplot as plt
#This code is for jupyter Notebook only
%matplotlib inline

# define data, and change list to array
x = [3,21,22,34,54,34,55,67,89,99]
x = np.array(x)
y = [2,22,24,65,79,82,55,130,150,199]
y = np.array(y)

#Show the effect of a scatter plot
plt.scatter(x,y)

```
![image](https://user-images.githubusercontent.com/114469025/192696093-76b22c47-3a92-4f6e-a3bf-a4615fe02063.png)

2. Определите связанные функции. Функция модели: определяет модель линейной регрессии wx+b. Функция потерь: функция потерь среднеквадратичной ошибки. Функция оптимизации: метод градиентного спуска для нахождения частных производных w и b.

```py

def model(a, b, x):
    return a * x + b
 
 
def loss_function(a, b, x, y):
    num = len(x)
    prediction = model(a, b, x)
    return (0.5 / num) * (np.square(prediction - y)).sum()
 
 
def optimize(a, b, x, y):
    num = len(x)
    prediction = model(a, b, x)
    da = (1.0 / num) * ((prediction - y) * x).sum()
    db = (1.0 / num) * ((prediction - y).sum())
    a = a - Lr * da
    b = b - Lr * db
    return a, b
 
 
def iterate(a, b, x, y, times):
    for i in range(times):
        a, b = optimize(a, b, x, y)
    return a, b


```
![image](https://user-images.githubusercontent.com/114469025/192696163-6746c586-e1a6-4127-9f12-5692ba04917a.png)

3. Начать итерацию.

- Шаг 1
```py

#Initialize parameters and display
a = np.random.rand(1)
print(a)
b = np.random.rand(1)
print(b)
Lr = 0.000001

#For the first iteration, the parameter values, losses, and visualization after the iteration are displayed
a,b = iterate(a,b,x,y,1)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)

```
![image](https://user-images.githubusercontent.com/114469025/192696222-df08493b-f4da-46b8-94e8-da52f78d63c7.png)

- Шаг 2
```py

a,b = iterate(a,b,x,y,2)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)

```
![image](https://user-images.githubusercontent.com/114469025/192696310-6f81c4bc-a9ba-40cd-afb6-fcf02e105419.png)

- Шаг 3
```py
a,b = iterate(a,b,x,y,3)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)

```
![image](https://user-images.githubusercontent.com/114469025/192696606-aac68fe9-632c-4168-93a7-9f9bea53b78b.png)

- Шаг 4
a,b = iterate(a,b,x,y,4)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)

```
![image](https://user-images.githubusercontent.com/114469025/192696752-b9bda522-5891-4356-a084-fc3d4a2e7b3e.png)

- Шаг 5
a,b = iterate(a,b,x,y,5)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)

```
![image](https://user-images.githubusercontent.com/114469025/192696827-1b9e380d-0556-477f-baf5-4bda346bb448.png)

- Шаг 6
a,b = iterate(a,b,x,y,10000)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)

```
![image](https://user-images.githubusercontent.com/114469025/192696901-9edafb34-51c4-4849-9ece-0439f4991134.png)


## Задание 3
### Должна ли величина loss стремиться к нулю при изменении исходных данных? Ответьте на вопрос, приведите пример выполнения кода, который подтверждает ваш ответ.

```py

```
### Какова роль параметра Lr? Ответьте на вопрос, приведите пример выполнения кода, который подтверждает ваш ответ. В качестве эксперимента можете изменить значение параметра.



## Выводы

Абзац умных слов о том, что было сделано и что было узнано.

| Plugin | README |
| ------ | ------ |
| Dropbox | [plugins/dropbox/README.md][PlDb] |
| GitHub | [plugins/github/README.md][PlGh] |
| Google Drive | [plugins/googledrive/README.md][PlGd] |
| OneDrive | [plugins/onedrive/README.md][PlOd] |
| Medium | [plugins/medium/README.md][PlMe] |
| Google Analytics | [plugins/googleanalytics/README.md][PlGa] |

## Powered by

**BigDigital Team: Denisov | Fadeev | Panov**
