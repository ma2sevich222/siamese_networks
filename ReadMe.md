# Siamese Neural Networks

## Science Articles:
- [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) and [code](https://github.com/fangpin/siamese-pytorch)
- [SigNet: Convolutional Siamese Network for Writer Independent Offline Signature Verification](https://arxiv.org/pdf/1707.02131.pdf) and [code](https://colab.research.google.com/drive/1KQgZkWVDOqywEV0lI6StVzIG84CCTpOe?usp=sharing)
- [Siamese Convolutional Neural Networks for Authorship Verification](http://cs231n.stanford.edu/reports/2017/pdfs/801.pdf)
- [Multi-Scale Contrastive Siamese Networks for Self-Supervised Graph Representation Learning](https://arxiv.org/pdf/2105.05682.pdf)
- [Similarity Learning based Few Shot Learning for ECG Time Series Classification](https://arxiv.org/pdf/2202.00612.pdf)
- [paperswithcode](https://paperswithcode.com/method/siamese-network)
- [другие научные статьи](https://arxiv.org/search/?query=Siamese+Networks&searchtype=all&source=header)



## Popular Literature:
- [Building image pairs for siamese networks with Python](https://www.pyimagesearch.com/2020/11/23/building-image-pairs-for-siamese-networks-with-python/)
- [Implementing Content-Based Image Retrieval With Siamese Networks in PyTorch](https://neptune.ai/blog/content-based-image-retrieval-with-siamese-networks)
- [PyTorch Practicing Project 4: Siamese Network](https://programmerall.com/article/1261218519/)
- [One Shot Learning with Siamese Networks in PyTorch](https://medium.com/hackernoon/one-shot-learning-with-siamese-networks-in-pytorch-8ddaab10340e)
- [Pytorch Siamese NN with BERT for sentence matching](https://stackoverflow.com/questions/66678360/pytorch-siamese-nn-with-bert-for-sentence-matching)
- [Building a One-shot Learning Network with PyTorch](https://towardsdatascience.com/building-a-one-shot-learning-network-with-pytorch-d1c3a5fafa4a)



## Codes:
- [24_tutorial.siamese.ipynb](https://colab.research.google.com/drive/166cGwY6ODauD156VCWde2WRGbjscMYvH?usp=sharing)
- [One Shot Learning with PyTorch](https://github.com/ttchengab/One_Shot_Pytorch)
- [Deep Neural Networks and Dynamic Factor Model on Time Series](https://github.innominds.com/ajayarunachalam/Deep_XF)
- [One Shot Learning with Siamese Networks using Keras](https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d)

## Loss:
- [COSINEEMBEDDINGLOSS](https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html)


## Инструкция:
1) **Файл constants.py.** Здесь задаются параметры данных для обработки файлов.
EXTR_WINDOW = 40 - параметр для отбора локальных экстремумов,если на отрезке [ i-40 : i+40]  значение индекса i является минимальным, оно будет являться локальным минимумом.
PATTERN_SIZE = 10 - размер паттерна. Определяет отрезок вида  [ i-10 : i ] где i  - индекс локального экстремума
START_TRAIN, END_TRAIN, START_TEST, END_TEST- отбираем данные для обучения и проверки соответственно
SOURCE_ROOT - Директория для сырых данных
DESTINATION_ROOT - Директория для сохранения результатов 
FILENAME - Имя файла с данными.

2)  **Файл train_test.py** Обучение и тестирование модели.Файл с результатами теста сохраняется в outputs.
Profit_value - процент от текущей цены. Дополнительное условие для отбора паттернов. Данная величина устанавливает процент от текущей цены, на величину которого будет рост на следующем отрезке [ i : i + EXTR_WINDOW ] где i - локальный минимум.
Ниже находятся закоментрированные функции дистанции, что бы можно было менять основную функцию дистанци для triplet loss. 


3) **pred_analysis.py** - выводид график предсказаний сети на тестовых данных. Вначале можно задать пределы дистанций 
для каждого вида предсказаний (всего из 3: цена будет рости, будет падать, не значительно измениться)


4) **model_tuning_run.py** - Запускает подбор гипер-параметров модели.



                                Схема работы файла train_test.py


<img src="/home/ma2sevich/Загрузки/схема(2).png" width="1560"/>
