# CIFAR-10 다운로드


```python
# 텐서플로(케라스) API를 사용
import numpy as np
import tensorflow as tf
from tensorflow import keras
# 텐서플로와 넘파이 확률 시작값을 고정
np.random.seed(1)
tf.random.set_seed(1)
```


```python
tf.__version__
```




    '2.4.1'




```python
label = ['비행기','자동차','새','고양이','사슴','개','개구리','말','배','트럭']
label_eng = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
```


```python
# 다운로드 받는다.
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
```

    Downloading data from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    170500096/170498071 [==============================] - 6s 0us/step
    


```python
# 32 * 32 * 3
X_train.shape, X_test.shape, y_train.shape, y_test.shape
```




    ((50000, 32, 32, 3), (10000, 32, 32, 3), (50000, 1), (10000, 1))




```python
X_train.ndim, X_test.ndim, y_train.ndim, y_test.ndim
```




    (4, 4, 2, 2)




```python
# 넘파이 배열에 중복값을 업애고 병합, return_counts: 입력배열의 등장횟수를 출력
np.unique(y_train, return_counts=True)
```




    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8),
     array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000]))




```python
label[6], label[9], y_train[0,0], y_train[1,0]
```




    ('개구리', '트럭', 6, 9)




```python
# 데이터 시각화(그래프) 출력
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
for i in range(0,9):
    plt.subplot(3,3, i+1)
    plt.imshow(X_train[i])  # X: 그림, y: 레이블
    plt.axis('off')
    plt.title(f'{y_train[i]}-{label_eng[y_train[i,0]]}')
plt.show()    
```


    
![png](output_9_0.png)
    


# 레이블된 훈련 데이터셋 재구성


```python
yt_idx = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]
rd_idx = np.random.choice(5000, size=500, replace=False) # 0~4999까지 랜덤 숫자 500개 1차원 넘파이배열 균등확률표본을 비복원추출
X500_train = np.zeros((5000,32,32,3), dtype=int)
y500_train = np.zeros((5000,1), dtype=int)      # 2차원 넘파이배열
delete_list=[]
for i in range(0,10): #[0]: # i: label 0 ~ 9
    # 클래스별 y_train의 인덱스를 반환, idx1: 0 ~ 4999, j: label 0 ~ 9
    i_idx = [idx1 for idx1, j in enumerate(y_train) if i==y_train[idx1,0] ] # 5만개에서 label[0~9]에 해당하는 인덱스만 list로 반환 => 5천개
    yt_idx[i] = i_idx # 10행 5000열 list 생성
    for idx2, nan in enumerate(rd_idx): # 500회 뽑기, idx2: 0~499, nan: 0~5000
        delete_list.append(yt_idx[i][nan]) # 500개의 원본 데이터셋 인덱스 저장
        X500_train[idx2+(i*500)] = X_train[yt_idx[i][nan]] # 자료형 (50000, 32, 32, 3)
        y500_train[idx2+(i*500)] = y_train[yt_idx[i][nan]] # 자료형 (50000, 1)
```


```python
X_train = np.delete(X_train, delete_list, axis=0 ) # X_train에서 사용된 데이터 (행)삭제
y_train = np.delete(y_train, delete_list, axis=0 ) # y_train에서 사용된 데이터 (행)삭제
```


```python
X500_train.shape, y500_train.shape
```




    ((5000, 32, 32, 3), (5000, 1))




```python
X500_train.ndim, y500_train.ndim
```




    (4, 2)




```python
print(label) # 클래스 당 500개의 데이터가 있는지 확인
label[y500_train[0,0]], label[y500_train[500,0]], label[y500_train[1000,0]], label[y500_train[1500,0]], label[y500_train[2000,0]], label[y500_train[2500,0]], label[y500_train[3000,0]], label[y500_train[3500,0]], label[y500_train[4000,0]], label[y500_train[4500,0]]
```

    ['비행기', '자동차', '새', '고양이', '사슴', '개', '개구리', '말', '배', '트럭']
    




    ('비행기', '자동차', '새', '고양이', '사슴', '개', '개구리', '말', '배', '트럭')



# 레이블된 테스트 데이터셋 재구성


```python
X50000_test = np.zeros((50000,32,32,3), dtype=int) # 4차원 넘파이배열
y50000_test = np.zeros((50000,1), dtype=int)       # 2차원 넘파이배열

yt_idx = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]
rd_idx = np.random.choice(1000, size=1000, replace=False) # 0~999까지 1000개 비복원추출 1차원 넘파이배열 균등확률표본을 비복원추출

for i in range(10): #[0]: # i: label 0 ~ 9
    # 클래스별 y_train의 인덱스를 반환, idx1: 0 ~ 999, j: label 0 ~ 9
    i_idx = [idx1 for idx1, j in enumerate(y_test) if i==y_test[idx1,0] ] # 1만개에서 label[0~9]에 해당하는 인덱스만 list로 반환
    yt_idx[i] = i_idx # 10행 1000열 list 생성
    for idx2, nan in enumerate(rd_idx): # 1000회 idx2: 0~999, nan: 0~999
        X50000_test[idx2+(i*1000)] = X_test[yt_idx[i][nan]] # 자료형 (50000, 32, 32, 3)
        y50000_test[idx2+(i*1000)] = y_test[yt_idx[i][nan]] # 자료형 (50000, 1)
        
yt_idx = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]
rd_idx = np.random.choice(4500, size=4000, replace=False) # 0~4499까지 랜덤 숫자 4000개 1차원 넘파이배열 균등확률표본을 비복원추출

for t in range(10): #[0]: # t: label 0 ~ 9
    # 클래스별 y_train의 인덱스를 반환, idx3: 0 ~ 44999, j: label 0 ~ 9
    i_idx1 = [idx3 for idx3, j in enumerate(y_train) if t==y_train[idx3,0] ] # 4만5천개에서 label[0~9]에 해당하는 인덱스만 list로 반환
    yt_idx[t] = i_idx1 # 10행 4500열 list 생성
    for idx4, nan1 in enumerate(rd_idx): # 4000회 idx2: 0~3999, nan: 0~4499 난수
        X50000_test[(idx4+10000)+t*4000] = X_train[yt_idx[t][nan1]] # 자료형 (45000, 32, 32, 3): X500_train 만들면서 5천개 삭감
        y50000_test[(idx4+10000)+t*4000] = y_train[yt_idx[t][nan1]] # 자료형 (45000, 1): y500_train 만들면서 5천개 삭감
```


```python
result = 0    # yt_idx[0][44999] 에러남. Class 당 4499개로 일정하지 않음
sum_list = [len(yt_idx[i]) for i in range(10)]
print(sum_list)
for idx, i in enumerate(sum_list):
    result = result + i
    if idx==9:
      mini = np.argmin(sum_list)
      print("총 데이터 합계: ", result)
      print("최소 원소 인덱스: ", mini, "\n최소 원소 개수: ", sum_list[8])
```

    [4496, 4522, 4483, 4489, 4534, 4530, 4488, 4479, 4475, 4504]
    총 데이터 합계:  45000
    최소 원소 인덱스:  8 
    최소 원소 개수:  4475
    


```python
X_train.shape, y_train.shape, X50000_test.shape, y50000_test.shape
```




    ((45000, 32, 32, 3), (45000, 1), (50000, 32, 32, 3), (50000, 1))




```python
X50000_test.ndim, y50000_test.ndim
```




    (4, 2)




```python
print(label)
# X_test, y_test 샘플링 데이터
label[y50000_test[0,0]], label[y50000_test[1000,0]],label[y50000_test[2000,0]],label[y50000_test[3000,0]],label[y50000_test[4000,0]],label[y50000_test[5000,0]],label[y50000_test[6000,0]],label[y50000_test[7000,0]],label[y50000_test[8000,0]],label[y50000_test[9000,0]]
```

    ['비행기', '자동차', '새', '고양이', '사슴', '개', '개구리', '말', '배', '트럭']
    




    ('비행기', '자동차', '새', '고양이', '사슴', '개', '개구리', '말', '배', '트럭')




```python
label[y50000_test[10000,0]], label[y50000_test[14000,0]], label[y50000_test[18000,0]], label[y50000_test[22000,0]], label[y50000_test[26000,0]], label[y50000_test[30000,0]], label[y50000_test[34000,0]], label[y50000_test[38000,0]], label[y50000_test[42000,0]], label[y50000_test[46000,0]]
```




    ('비행기', '자동차', '새', '고양이', '사슴', '개', '개구리', '말', '배', '트럭')




```python
# 데이터를 시각화(그래프) 출력
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
for i in range(0,9):
    plt.subplot(3,3, i+1)   # 3행3열의 여러개 그림
    plt.imshow(X50000_test[i])  # X: 그림, y: 레이블
    plt.axis('off')
    plt.title(f'{y50000_test[i]}-{label_eng[y50000_test[i,0]]}')
plt.show() 
```


    
![png](output_23_0.png)
    


# ImageDataGenerator생성


```python
# hyper parameter
learning_rate = 0.002  # 0.003
n_epochs = 80   # 60
n_batchs = 348         # 128
n_class = 10
dropout_rate = 0.5     # 0.3
```


```python
# y - onehot encoding
from tensorflow.keras.utils import to_categorical
y500_train = keras.utils.to_categorical(y500_train, num_classes=n_class)
y50000_test = keras.utils.to_categorical(y50000_test, num_classes=n_class)
```


```python
y500_train.shape
```




    (5000, 10)



# 이미지 증강 테스트


```python
# 이미지 증강을 위한 모듈과 메소드
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_gen = ImageDataGenerator(
    rescale=1/255.,
    rotation_range=40, # 30
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
    #fill_mode='nearest')

val_gen = ImageDataGenerator(rescale=1/255.)
```


```python
# DataGenerator.flow(): 메모리의 배열로 부터 읽어서 생성, 
train_generator = train_gen.flow(X500_train, y500_train, batch_size=n_batchs)
val_generator = val_gen.flow(X50000_test, y50000_test, batch_size=n_batchs)
# print(n_batchs)
```


```python
train_generator[0][0][0].shape
```




    (32, 32, 3)




```python
# 구글 드라이브 마운트
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive
    


```python
%matplotlib inline
# 운영체제에서 제공되는 여러 기능을 파이썬에서 수행
import os
# 데이터를 시각화(그래프) 출력에 필요한 모듈
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# 이미지를 업로드하고 넘파이 배열로 변환 모듈과 메소드
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img

curr_dir = os.getcwd() # current work director
print(curr_dir)
img_path = "/content/drive/MyDrive/Colab Notebooks/deer.jpg"
img1 = load_img(img_path, target_size=(32, 32))  # this is a PIL image: PIL(Python Image Library)
# img
x1 = img_to_array(img1)  # Numpy array with shape (32, 32, 3)
x1 = x1.reshape((1,) + x1.shape)  # Numpy array with shape (1, 32, 32, 3)
print(x1.shape)
```

    /content
    (1, 32, 32, 3)
    


```python
plt.figure(figsize=(10, 10))
i=0
for batch in train_gen.flow(x1, batch_size=1):
  plt.subplot(3, 3, i + 1)
  plt.imshow(array_to_img(batch[0]))
  plt.axis("off")
  plt.title(f'deer-test[{i}]')
  i += 1
  if i % 9 == 0:
    break
plt.show()
```


    
![png](output_34_0.png)
    



```python
print('훈련세트', X500_train_centered.shape, y500_train.shape)
print('테스트 세트', X50000_test_centered.shape, y50000_test.shape)
```

    훈련세트 (5000, 32, 32, 3) (5000, 10)
    테스트 세트 (50000, 32, 32, 3) (50000, 10)
    

# 모델 구성


```python
# 텐서플로 인공신경망 구성을 위한 모듈
from tensorflow.keras import layers, models
def create_model():
    model = models.Sequential()
    #Conv Layer
    model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='SAME', input_shape=(32,32,3)))
    model.add(layers.MaxPool2D((2,2))) # padding='valid' (default):유효한 영역만 출력, 'same': 출력 이미지 사이즈가 입력 이미지 사이즈와 동일
    model.add(layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='SAME'))
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='SAME'))
    model.add(layers.MaxPool2D((2,2)))

    # Full Connected Layer
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu')) # 128
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(10, activation='softmax'))
    return model
```


```python
model = create_model()
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 32, 32, 64)        1792      
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 16, 16, 64)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 16, 16, 128)       73856     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 8, 8, 128)         0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 8, 8, 128)         147584    
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 4, 4, 128)         0         
    _________________________________________________________________
    flatten (Flatten)            (None, 2048)              0         
    _________________________________________________________________
    dense (Dense)                (None, 1024)              2098176   
    _________________________________________________________________
    dropout (Dropout)            (None, 1024)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                10250     
    =================================================================
    Total params: 2,331,658
    Trainable params: 2,331,658
    Non-trainable params: 0
    _________________________________________________________________
    


```python
# from keras.utils import plot_model
# plot_model(model, to_file='model_shapes.png', show_shapes=True)
```

# 분류모델 훈련


```python
# 학습 전에 테스트
model.evaluate(val_generator)
```

    144/144 [==============================] - 6s 9ms/step - loss: 2.3017 - accuracy: 0.1007
    




    [2.3014628887176514, 0.1016400009393692]




```python
# 운영체제에서 제공되는 여러 기능을 파이썬에서 수행
import os
root_path = '/content/drive/My Drive/save_models'
save_path = os.path.join(root_path, 'cifar10_model_imgaug_data1')
save_path
```




    '/content/drive/My Drive/save_models/cifar10_model_imgaug_data1'




```python
# 시간 모듈
import time
start = time.time()
modelcheckpoint_callback = keras.callbacks.ModelCheckpoint(filepath=save_path, monitor='val_loss', save_best_only=True)


history = model.fit(train_generator, 
                    epochs=n_epochs,
                    steps_per_epoch=len(train_generator),
                    validation_data=val_generator,
                    validation_steps=len(val_generator),
                    callbacks=[modelcheckpoint_callback]
                    )
end = time.time()
```

    Epoch 1/100
    15/15 [==============================] - 5s 283ms/step - loss: 2.2423 - accuracy: 0.1488 - val_loss: 2.0742 - val_accuracy: 0.2351
    INFO:tensorflow:Assets written to: /content/drive/My Drive/save_models/cifar10_model_imgaug_data1/assets
    Epoch 2/100
    15/15 [==============================] - 4s 264ms/step - loss: 2.0682 - accuracy: 0.2414 - val_loss: 1.9559 - val_accuracy: 0.2891
    INFO:tensorflow:Assets written to: /content/drive/My Drive/save_models/cifar10_model_imgaug_data1/assets
    Epoch 3/100
    15/15 [==============================] - 4s 260ms/step - loss: 1.9759 - accuracy: 0.2686 - val_loss: 1.8656 - val_accuracy: 0.3226
    INFO:tensorflow:Assets written to: /content/drive/My Drive/save_models/cifar10_model_imgaug_data1/assets
    Epoch 4/100
    15/15 [==============================] - 4s 260ms/step - loss: 1.9045 - accuracy: 0.3062 - val_loss: 1.7483 - val_accuracy: 0.3752
    INFO:tensorflow:Assets written to: /content/drive/My Drive/save_models/cifar10_model_imgaug_data1/assets
    Epoch 5/100
    15/15 [==============================] - 4s 261ms/step - loss: 1.8315 - accuracy: 0.3314 - val_loss: 1.7130 - val_accuracy: 0.3859
    INFO:tensorflow:Assets written to: /content/drive/My Drive/save_models/cifar10_model_imgaug_data1/assets
    Epoch 6/100
    15/15 [==============================] - 4s 262ms/step - loss: 1.7888 - accuracy: 0.3346 - val_loss: 1.6572 - val_accuracy: 0.4016
    INFO:tensorflow:Assets written to: /content/drive/My Drive/save_models/cifar10_model_imgaug_data1/assets
    Epoch 7/100
    15/15 [==============================] - 4s 257ms/step - loss: 1.7648 - accuracy: 0.3590 - val_loss: 1.6162 - val_accuracy: 0.4086
    INFO:tensorflow:Assets written to: /content/drive/My Drive/save_models/cifar10_model_imgaug_data1/assets
    Epoch 8/100
    15/15 [==============================] - 4s 259ms/step - loss: 1.7355 - accuracy: 0.3738 - val_loss: 1.7052 - val_accuracy: 0.3959
    Epoch 9/100
    15/15 [==============================] - 4s 264ms/step - loss: 1.7305 - accuracy: 0.3798 - val_loss: 1.6181 - val_accuracy: 0.4159
    Epoch 10/100
    15/15 [==============================] - 4s 266ms/step - loss: 1.6875 - accuracy: 0.3836 - val_loss: 1.6227 - val_accuracy: 0.4117
    Epoch 11/100
    15/15 [==============================] - 4s 261ms/step - loss: 1.6758 - accuracy: 0.3972 - val_loss: 1.6954 - val_accuracy: 0.3804
    Epoch 12/100
    15/15 [==============================] - 4s 258ms/step - loss: 1.6340 - accuracy: 0.4060 - val_loss: 1.4844 - val_accuracy: 0.4527
    INFO:tensorflow:Assets written to: /content/drive/My Drive/save_models/cifar10_model_imgaug_data1/assets
    Epoch 13/100
    15/15 [==============================] - 4s 263ms/step - loss: 1.5970 - accuracy: 0.4232 - val_loss: 1.5245 - val_accuracy: 0.4489
    Epoch 14/100
    15/15 [==============================] - 4s 269ms/step - loss: 1.5632 - accuracy: 0.4296 - val_loss: 1.4391 - val_accuracy: 0.4775
    INFO:tensorflow:Assets written to: /content/drive/My Drive/save_models/cifar10_model_imgaug_data1/assets
    Epoch 15/100
    15/15 [==============================] - 4s 267ms/step - loss: 1.5381 - accuracy: 0.4404 - val_loss: 1.4101 - val_accuracy: 0.4871
    INFO:tensorflow:Assets written to: /content/drive/My Drive/save_models/cifar10_model_imgaug_data1/assets
    Epoch 16/100
    15/15 [==============================] - 4s 260ms/step - loss: 1.5433 - accuracy: 0.4524 - val_loss: 1.4057 - val_accuracy: 0.4865
    INFO:tensorflow:Assets written to: /content/drive/My Drive/save_models/cifar10_model_imgaug_data1/assets
    Epoch 17/100
    15/15 [==============================] - 4s 260ms/step - loss: 1.5121 - accuracy: 0.4600 - val_loss: 1.4889 - val_accuracy: 0.4699
    Epoch 18/100
    15/15 [==============================] - 4s 263ms/step - loss: 1.5145 - accuracy: 0.4500 - val_loss: 1.3953 - val_accuracy: 0.4930
    INFO:tensorflow:Assets written to: /content/drive/My Drive/save_models/cifar10_model_imgaug_data1/assets
    Epoch 19/100
    15/15 [==============================] - 4s 260ms/step - loss: 1.4932 - accuracy: 0.4610 - val_loss: 1.4522 - val_accuracy: 0.4749
    Epoch 20/100
    15/15 [==============================] - 4s 261ms/step - loss: 1.4657 - accuracy: 0.4738 - val_loss: 1.4209 - val_accuracy: 0.4947
    Epoch 21/100
    15/15 [==============================] - 4s 264ms/step - loss: 1.4517 - accuracy: 0.4720 - val_loss: 1.3941 - val_accuracy: 0.5031
    INFO:tensorflow:Assets written to: /content/drive/My Drive/save_models/cifar10_model_imgaug_data1/assets
    Epoch 22/100
    15/15 [==============================] - 4s 257ms/step - loss: 1.4271 - accuracy: 0.4868 - val_loss: 1.3303 - val_accuracy: 0.5208
    INFO:tensorflow:Assets written to: /content/drive/My Drive/save_models/cifar10_model_imgaug_data1/assets
    Epoch 23/100
    15/15 [==============================] - 4s 259ms/step - loss: 1.4212 - accuracy: 0.4862 - val_loss: 1.3587 - val_accuracy: 0.5105
    Epoch 24/100
    15/15 [==============================] - 4s 267ms/step - loss: 1.4002 - accuracy: 0.5036 - val_loss: 1.3115 - val_accuracy: 0.5293
    INFO:tensorflow:Assets written to: /content/drive/My Drive/save_models/cifar10_model_imgaug_data1/assets
    Epoch 25/100
    15/15 [==============================] - 4s 266ms/step - loss: 1.4167 - accuracy: 0.4916 - val_loss: 1.4272 - val_accuracy: 0.4988
    Epoch 26/100
    15/15 [==============================] - 4s 261ms/step - loss: 1.3917 - accuracy: 0.4962 - val_loss: 1.2825 - val_accuracy: 0.5383
    INFO:tensorflow:Assets written to: /content/drive/My Drive/save_models/cifar10_model_imgaug_data1/assets
    Epoch 27/100
    15/15 [==============================] - 4s 259ms/step - loss: 1.3829 - accuracy: 0.5012 - val_loss: 1.3957 - val_accuracy: 0.5001
    Epoch 28/100
    15/15 [==============================] - 4s 259ms/step - loss: 1.3381 - accuracy: 0.5244 - val_loss: 1.3444 - val_accuracy: 0.5306
    Epoch 29/100
    15/15 [==============================] - 4s 260ms/step - loss: 1.3566 - accuracy: 0.5172 - val_loss: 1.3205 - val_accuracy: 0.5283
    Epoch 30/100
    15/15 [==============================] - 4s 268ms/step - loss: 1.3774 - accuracy: 0.5122 - val_loss: 1.2806 - val_accuracy: 0.5388
    INFO:tensorflow:Assets written to: /content/drive/My Drive/save_models/cifar10_model_imgaug_data1/assets
    Epoch 31/100
    15/15 [==============================] - 4s 258ms/step - loss: 1.3065 - accuracy: 0.5300 - val_loss: 1.3475 - val_accuracy: 0.5236
    Epoch 32/100
    15/15 [==============================] - 4s 262ms/step - loss: 1.3072 - accuracy: 0.5258 - val_loss: 1.2513 - val_accuracy: 0.5501
    INFO:tensorflow:Assets written to: /content/drive/My Drive/save_models/cifar10_model_imgaug_data1/assets
    Epoch 33/100
    15/15 [==============================] - 4s 261ms/step - loss: 1.3075 - accuracy: 0.5296 - val_loss: 1.2740 - val_accuracy: 0.5478
    Epoch 34/100
    15/15 [==============================] - 4s 271ms/step - loss: 1.3151 - accuracy: 0.5246 - val_loss: 1.2972 - val_accuracy: 0.5411
    Epoch 35/100
    15/15 [==============================] - 4s 261ms/step - loss: 1.3020 - accuracy: 0.5266 - val_loss: 1.2755 - val_accuracy: 0.5505
    Epoch 36/100
    15/15 [==============================] - 4s 260ms/step - loss: 1.2634 - accuracy: 0.5508 - val_loss: 1.2272 - val_accuracy: 0.5624
    INFO:tensorflow:Assets written to: /content/drive/My Drive/save_models/cifar10_model_imgaug_data1/assets
    Epoch 37/100
    15/15 [==============================] - 4s 263ms/step - loss: 1.2584 - accuracy: 0.5520 - val_loss: 1.3184 - val_accuracy: 0.5414
    Epoch 38/100
    15/15 [==============================] - 4s 270ms/step - loss: 1.2578 - accuracy: 0.5484 - val_loss: 1.2218 - val_accuracy: 0.5732
    INFO:tensorflow:Assets written to: /content/drive/My Drive/save_models/cifar10_model_imgaug_data1/assets
    Epoch 39/100
    15/15 [==============================] - 4s 259ms/step - loss: 1.2598 - accuracy: 0.5458 - val_loss: 1.3228 - val_accuracy: 0.5370
    Epoch 40/100
    15/15 [==============================] - 4s 260ms/step - loss: 1.2358 - accuracy: 0.5612 - val_loss: 1.2479 - val_accuracy: 0.5686
    Epoch 41/100
    15/15 [==============================] - 4s 259ms/step - loss: 1.2426 - accuracy: 0.5518 - val_loss: 1.2462 - val_accuracy: 0.5632
    Epoch 42/100
    15/15 [==============================] - 4s 260ms/step - loss: 1.2132 - accuracy: 0.5600 - val_loss: 1.2665 - val_accuracy: 0.5584
    Epoch 43/100
    15/15 [==============================] - 4s 259ms/step - loss: 1.2031 - accuracy: 0.5714 - val_loss: 1.2309 - val_accuracy: 0.5739
    Epoch 44/100
    15/15 [==============================] - 4s 258ms/step - loss: 1.1900 - accuracy: 0.5714 - val_loss: 1.1813 - val_accuracy: 0.5833
    INFO:tensorflow:Assets written to: /content/drive/My Drive/save_models/cifar10_model_imgaug_data1/assets
    Epoch 45/100
    15/15 [==============================] - 4s 260ms/step - loss: 1.1579 - accuracy: 0.5880 - val_loss: 1.2020 - val_accuracy: 0.5815
    Epoch 46/100
    15/15 [==============================] - 4s 265ms/step - loss: 1.1883 - accuracy: 0.5864 - val_loss: 1.2337 - val_accuracy: 0.5670
    Epoch 47/100
    15/15 [==============================] - 4s 267ms/step - loss: 1.1481 - accuracy: 0.5898 - val_loss: 1.1734 - val_accuracy: 0.5907
    INFO:tensorflow:Assets written to: /content/drive/My Drive/save_models/cifar10_model_imgaug_data1/assets
    Epoch 48/100
    15/15 [==============================] - 4s 260ms/step - loss: 1.1409 - accuracy: 0.5908 - val_loss: 1.1090 - val_accuracy: 0.6110
    INFO:tensorflow:Assets written to: /content/drive/My Drive/save_models/cifar10_model_imgaug_data1/assets
    Epoch 49/100
    15/15 [==============================] - 4s 261ms/step - loss: 1.1510 - accuracy: 0.5864 - val_loss: 1.2239 - val_accuracy: 0.5743
    Epoch 50/100
    15/15 [==============================] - 4s 265ms/step - loss: 1.1274 - accuracy: 0.5962 - val_loss: 1.1302 - val_accuracy: 0.6075
    Epoch 51/100
    15/15 [==============================] - 4s 262ms/step - loss: 1.1366 - accuracy: 0.5910 - val_loss: 1.1727 - val_accuracy: 0.5944
    Epoch 52/100
    15/15 [==============================] - 4s 261ms/step - loss: 1.1394 - accuracy: 0.5906 - val_loss: 1.2456 - val_accuracy: 0.5756
    Epoch 53/100
    15/15 [==============================] - 4s 260ms/step - loss: 1.1354 - accuracy: 0.6012 - val_loss: 1.1213 - val_accuracy: 0.6075
    Epoch 54/100
    15/15 [==============================] - 4s 260ms/step - loss: 1.1037 - accuracy: 0.6030 - val_loss: 1.1719 - val_accuracy: 0.5974
    Epoch 55/100
    15/15 [==============================] - 4s 258ms/step - loss: 1.1089 - accuracy: 0.6098 - val_loss: 1.1141 - val_accuracy: 0.6123
    Epoch 56/100
    15/15 [==============================] - 4s 263ms/step - loss: 1.0951 - accuracy: 0.6058 - val_loss: 1.2356 - val_accuracy: 0.5818
    Epoch 57/100
    15/15 [==============================] - 4s 262ms/step - loss: 1.0715 - accuracy: 0.6200 - val_loss: 1.1483 - val_accuracy: 0.6058
    Epoch 58/100
    15/15 [==============================] - 4s 260ms/step - loss: 1.0647 - accuracy: 0.6156 - val_loss: 1.2325 - val_accuracy: 0.5836
    Epoch 59/100
    15/15 [==============================] - 4s 261ms/step - loss: 1.1058 - accuracy: 0.6120 - val_loss: 1.2627 - val_accuracy: 0.5619
    Epoch 60/100
    15/15 [==============================] - 4s 260ms/step - loss: 1.1076 - accuracy: 0.6122 - val_loss: 1.1448 - val_accuracy: 0.6041
    Epoch 61/100
    15/15 [==============================] - 4s 265ms/step - loss: 1.0503 - accuracy: 0.6336 - val_loss: 1.1724 - val_accuracy: 0.6027
    Epoch 62/100
    15/15 [==============================] - 4s 263ms/step - loss: 1.0696 - accuracy: 0.6220 - val_loss: 1.0505 - val_accuracy: 0.6307
    INFO:tensorflow:Assets written to: /content/drive/My Drive/save_models/cifar10_model_imgaug_data1/assets
    Epoch 63/100
    15/15 [==============================] - 4s 265ms/step - loss: 1.0338 - accuracy: 0.6340 - val_loss: 1.0992 - val_accuracy: 0.6206
    Epoch 64/100
    15/15 [==============================] - 4s 262ms/step - loss: 1.0215 - accuracy: 0.6326 - val_loss: 1.0725 - val_accuracy: 0.6266
    Epoch 65/100
    15/15 [==============================] - 4s 263ms/step - loss: 1.0485 - accuracy: 0.6296 - val_loss: 1.0809 - val_accuracy: 0.6202
    Epoch 66/100
    15/15 [==============================] - 4s 268ms/step - loss: 1.0655 - accuracy: 0.6188 - val_loss: 1.1136 - val_accuracy: 0.6162
    Epoch 67/100
    15/15 [==============================] - 4s 260ms/step - loss: 1.0342 - accuracy: 0.6256 - val_loss: 1.0810 - val_accuracy: 0.6238
    Epoch 68/100
    15/15 [==============================] - 4s 260ms/step - loss: 1.0118 - accuracy: 0.6404 - val_loss: 1.1949 - val_accuracy: 0.5937
    Epoch 69/100
    15/15 [==============================] - 4s 269ms/step - loss: 0.9991 - accuracy: 0.6446 - val_loss: 1.0763 - val_accuracy: 0.6281
    Epoch 70/100
    15/15 [==============================] - 4s 259ms/step - loss: 1.0116 - accuracy: 0.6450 - val_loss: 1.1269 - val_accuracy: 0.6141
    Epoch 71/100
    15/15 [==============================] - 4s 260ms/step - loss: 1.0130 - accuracy: 0.6414 - val_loss: 1.0614 - val_accuracy: 0.6329
    Epoch 72/100
    15/15 [==============================] - 4s 264ms/step - loss: 0.9965 - accuracy: 0.6400 - val_loss: 1.1089 - val_accuracy: 0.6211
    Epoch 73/100
    15/15 [==============================] - 4s 263ms/step - loss: 0.9878 - accuracy: 0.6470 - val_loss: 1.1080 - val_accuracy: 0.6220
    Epoch 74/100
    15/15 [==============================] - 4s 274ms/step - loss: 0.9716 - accuracy: 0.6578 - val_loss: 1.1445 - val_accuracy: 0.6100
    Epoch 75/100
    15/15 [==============================] - 4s 260ms/step - loss: 0.9523 - accuracy: 0.6614 - val_loss: 1.1982 - val_accuracy: 0.6046
    Epoch 76/100
    15/15 [==============================] - 4s 260ms/step - loss: 0.9846 - accuracy: 0.6480 - val_loss: 1.0769 - val_accuracy: 0.6304
    Epoch 77/100
    15/15 [==============================] - 4s 258ms/step - loss: 0.9748 - accuracy: 0.6606 - val_loss: 1.1171 - val_accuracy: 0.6183
    Epoch 78/100
    15/15 [==============================] - 4s 263ms/step - loss: 0.9519 - accuracy: 0.6586 - val_loss: 1.1242 - val_accuracy: 0.6218
    Epoch 79/100
    15/15 [==============================] - 4s 258ms/step - loss: 0.9327 - accuracy: 0.6684 - val_loss: 1.1034 - val_accuracy: 0.6256
    Epoch 80/100
    15/15 [==============================] - 4s 268ms/step - loss: 0.9334 - accuracy: 0.6684 - val_loss: 1.0614 - val_accuracy: 0.6405
    Epoch 81/100
    15/15 [==============================] - 4s 259ms/step - loss: 0.9324 - accuracy: 0.6738 - val_loss: 1.1035 - val_accuracy: 0.6289
    Epoch 82/100
    15/15 [==============================] - 4s 262ms/step - loss: 0.9265 - accuracy: 0.6748 - val_loss: 1.1082 - val_accuracy: 0.6244
    Epoch 83/100
    15/15 [==============================] - 4s 262ms/step - loss: 0.9519 - accuracy: 0.6626 - val_loss: 1.1705 - val_accuracy: 0.6132
    Epoch 84/100
    15/15 [==============================] - 4s 264ms/step - loss: 0.9348 - accuracy: 0.6686 - val_loss: 1.0450 - val_accuracy: 0.6400
    INFO:tensorflow:Assets written to: /content/drive/My Drive/save_models/cifar10_model_imgaug_data1/assets
    Epoch 85/100
    15/15 [==============================] - 4s 261ms/step - loss: 0.9191 - accuracy: 0.6812 - val_loss: 1.1531 - val_accuracy: 0.6145
    Epoch 86/100
    15/15 [==============================] - 4s 259ms/step - loss: 0.9158 - accuracy: 0.6760 - val_loss: 1.0782 - val_accuracy: 0.6384
    Epoch 87/100
    15/15 [==============================] - 4s 262ms/step - loss: 0.8957 - accuracy: 0.6874 - val_loss: 1.2014 - val_accuracy: 0.6102
    Epoch 88/100
    15/15 [==============================] - 4s 258ms/step - loss: 0.8973 - accuracy: 0.6780 - val_loss: 1.0950 - val_accuracy: 0.6290
    Epoch 89/100
    15/15 [==============================] - 4s 261ms/step - loss: 0.8996 - accuracy: 0.6770 - val_loss: 1.1623 - val_accuracy: 0.6189
    Epoch 90/100
    15/15 [==============================] - 4s 269ms/step - loss: 0.9188 - accuracy: 0.6766 - val_loss: 1.0592 - val_accuracy: 0.6402
    Epoch 91/100
    15/15 [==============================] - 4s 267ms/step - loss: 0.8649 - accuracy: 0.6822 - val_loss: 1.1361 - val_accuracy: 0.6261
    Epoch 92/100
    15/15 [==============================] - 4s 261ms/step - loss: 0.8623 - accuracy: 0.6862 - val_loss: 1.1149 - val_accuracy: 0.6305
    Epoch 93/100
    15/15 [==============================] - 4s 262ms/step - loss: 0.8692 - accuracy: 0.6952 - val_loss: 1.1784 - val_accuracy: 0.6194
    Epoch 94/100
    15/15 [==============================] - 4s 268ms/step - loss: 0.8888 - accuracy: 0.6822 - val_loss: 1.0510 - val_accuracy: 0.6408
    Epoch 95/100
    15/15 [==============================] - 4s 259ms/step - loss: 0.8734 - accuracy: 0.6908 - val_loss: 1.0936 - val_accuracy: 0.6361
    Epoch 96/100
    15/15 [==============================] - 4s 259ms/step - loss: 0.8480 - accuracy: 0.6902 - val_loss: 1.0905 - val_accuracy: 0.6421
    Epoch 97/100
    15/15 [==============================] - 4s 258ms/step - loss: 0.8500 - accuracy: 0.6944 - val_loss: 1.0679 - val_accuracy: 0.6492
    Epoch 98/100
    15/15 [==============================] - 4s 260ms/step - loss: 0.9113 - accuracy: 0.6714 - val_loss: 1.0267 - val_accuracy: 0.6510
    INFO:tensorflow:Assets written to: /content/drive/My Drive/save_models/cifar10_model_imgaug_data1/assets
    Epoch 99/100
    15/15 [==============================] - 4s 262ms/step - loss: 0.8797 - accuracy: 0.6862 - val_loss: 1.1543 - val_accuracy: 0.6211
    Epoch 100/100
    15/15 [==============================] - 4s 271ms/step - loss: 0.8404 - accuracy: 0.7028 - val_loss: 1.1701 - val_accuracy: 0.6220
    

# 모델 성능 테스트 결과


```python
print((end-start)/60, '분') # 6.39 분
```

    6.912581566969553 분
    


```python
saved_model = keras.models.load_model(save_path)
```


```python
saved_model.evaluate(val_generator) # loss: 1.0812 - accuracy: 0.6331
```

    144/144 [==============================] - 1s 9ms/step - loss: 1.0267 - accuracy: 0.6510
    




    [1.0267202854156494, 0.6509799957275391]




```python
history.history.keys()
```




    dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])




```python
n_epochs
```




    100




```python
# 데이터 시각화(그래프) 출력 모듈
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.plot(range(n_epochs), history.history['loss'], label='train loss')
plt.plot(range(n_epochs), history.history['val_loss'], label='val loss')
plt.legend()
plt.title('loss')
plt.show()
```


    
![png](output_50_0.png)
    



```python
# 데이터 시각화(그래프) 출력 모듈
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.title('accuracy')
plt.show()
```


    
![png](output_51_0.png)
    


# 새로운 이미지 분류(예측)


```python
# 운영체제에서 제공되는 여러 기능을 파이썬에서 수행
import os
curr_dir = os.getcwd() # current work director
print(curr_dir)
# x특정 디렉토리 안에 있는 하위티렉토리나 파일들의 이름을 리스트에 담아서 반환.
# path를 지정하거나 생략하면 현재디렉토리 내의 내용을 반환.
path_list = os.listdir()
print(path_list)
```

    /content
    ['.config', 'drive', 'sample_data']
    


```python
# 이미지를 업로드하고 넘파이 배열로 변환하는 모듈과 메소드
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# my_img_path = "/content/truck1.jpg" 
my_img_path = "/content/drive/MyDrive/Colab Notebooks/frog1.jpg"
img = load_img(my_img_path, target_size=(32, 32))
img
```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    <ipython-input-1-b69ca4a8322f> in <module>()
          3 # my_img_path = "/content/truck1.jpg"
          4 my_img_path = "/content/drive/MyDrive/Colab Notebooks/frog1.jpg"
    ----> 5 img = load_img(my_img_path, target_size=(32, 32))
          6 img
    

    /usr/local/lib/python3.6/dist-packages/tensorflow/python/keras/preprocessing/image.py in load_img(path, grayscale, color_mode, target_size, interpolation)
        298   """
        299   return image.load_img(path, grayscale=grayscale, color_mode=color_mode,
    --> 300                         target_size=target_size, interpolation=interpolation)
        301 
        302 
    

    /usr/local/lib/python3.6/dist-packages/keras_preprocessing/image/utils.py in load_img(path, grayscale, color_mode, target_size, interpolation)
        111         raise ImportError('Could not import PIL.Image. '
        112                           'The use of `load_img` requires PIL.')
    --> 113     with open(path, 'rb') as f:
        114         img = pil_image.open(io.BytesIO(f.read()))
        115         if color_mode == 'grayscale':
    

    FileNotFoundError: [Errno 2] No such file or directory: '/content/drive/MyDrive/Colab Notebooks/frog1.jpg'



```python
# 내부경로로부터 이미지를 업로드하여 훈련된 모델로 이미지의 레이블 예측
x = img_to_array(img) #ndarray변환
x = x[np.newaxis, ...] # 0번축 추가 
sample_x = x/255.  #정규화 
pred = saved_model.predict(sample_x) # 10개의 예측값 숫자배열
cls = np.argmax(pred, axis=-1)  # 가장 높은 신뢰도의 레이블
pred, cls
```




    (array([[1.0414654e-02, 1.6308134e-02, 3.1529597e-04, 5.5251759e-09,
             1.8217041e-05, 2.0582858e-08, 9.7291493e-01, 3.1076613e-08,
             2.2263031e-07, 2.8552215e-05]], dtype=float32), array([6]))




```python
pred[0, cls[0]], label[cls[0]]
```




    (0.97291493, '개구리')




```python
# 이미지를 업로드하고 넘파이 배열로 변환하는 모듈과 메소드
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# my_img_path = "/content/truck1.jpg" 
my_img_path = "/content/drive/MyDrive/Colab Notebooks/deer.jpg"
img = load_img(my_img_path, target_size=(32, 32))
img

```




    
![png](output_57_0.png)
    




```python
# 내부경로로부터 이미지를 업로드하여 훈련된 모델로 이미지의 레이블 예측
x1 = img_to_array(img) #ndarray변환
x1 = x1[np.newaxis, ...] # 0번축 추가 
sample_x1 = x1/255.  #정규화 
pred = saved_model.predict(sample_x1)
cls = np.argmax(pred, axis=-1)
pred, cls
```




    (array([[0.00967897, 0.00399721, 0.28603584, 0.16213097, 0.17227118,
             0.13445757, 0.182078  , 0.01333272, 0.035439  , 0.00057861]],
           dtype=float32), array([2]))




```python
pred[0, cls[0]], label[cls[0]]
```




    (0.28603584, '새')




```python

```
