

```python
from sklearn.cluster import KMeans 
```


```python
import numpy as np
import pandas as pd
import seaborn as sb #데이터 시각화를 위한 seaborn
import matplotlib.pyplot as plt
%matplotlib inline #notebook을 실행한 브라우저에서 그림을 볼 수 있게 한다.출력에 관함.
```


```python
df = pd.DataFrame(columns=['x','y'])
```


```python
df.loc[0] = [2,3]
df.loc[1] = [12,3]
df.loc[2] = [24,33]
df.loc[3] = [52,13]
df.loc[4] = [22,30]
df.loc[5] = [62,83]
df.loc[6] = [82,3]
df.loc[7] = [72,38]
df.loc[8] = [92,30]
df.loc[9] = [82,93]
df.loc[10] = [28,33]
df.loc[11] = [23,23]
df.loc[12] = [62,38]
df.loc[13] = [22,23]
df.loc[14] = [52,13]
df.loc[15] = [24,37]
df.loc[16] = [52,13]
df.loc[17] = [26,93]
df.loc[18] = [62,83]
df.loc[19] = [12,35]
df.loc[20] = [22,39]
```


```python
df.head(20) #표형태로 출력
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>24</td>
      <td>33</td>
    </tr>
    <tr>
      <th>3</th>
      <td>52</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22</td>
      <td>30</td>
    </tr>
    <tr>
      <th>5</th>
      <td>62</td>
      <td>83</td>
    </tr>
    <tr>
      <th>6</th>
      <td>82</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>72</td>
      <td>38</td>
    </tr>
    <tr>
      <th>8</th>
      <td>92</td>
      <td>30</td>
    </tr>
    <tr>
      <th>9</th>
      <td>82</td>
      <td>93</td>
    </tr>
    <tr>
      <th>10</th>
      <td>28</td>
      <td>33</td>
    </tr>
    <tr>
      <th>11</th>
      <td>23</td>
      <td>23</td>
    </tr>
    <tr>
      <th>12</th>
      <td>62</td>
      <td>38</td>
    </tr>
    <tr>
      <th>13</th>
      <td>22</td>
      <td>23</td>
    </tr>
    <tr>
      <th>14</th>
      <td>52</td>
      <td>13</td>
    </tr>
    <tr>
      <th>15</th>
      <td>24</td>
      <td>37</td>
    </tr>
    <tr>
      <th>16</th>
      <td>52</td>
      <td>13</td>
    </tr>
    <tr>
      <th>17</th>
      <td>26</td>
      <td>93</td>
    </tr>
    <tr>
      <th>18</th>
      <td>62</td>
      <td>83</td>
    </tr>
    <tr>
      <th>19</th>
      <td>12</td>
      <td>35</td>
    </tr>
    <tr>
      <th>20</th>
      <td>22</td>
      <td>39</td>
    </tr>
  </tbody>
</table>
</div>




```python
sb.lmplot('x','y',data=df, fit_reg=False, scatter_kws={"s": 100}) #좌표평면에 출력
plt.title('k-means example')
plt.xlabel('x축')
plt.ylabel('y축')
```




    Text(10.05,0.5,'y축')




![png](output_5_1.png)



```python
points = df.values #dataFrame의 값들을 numpy객체로서 초기화해준다.
kmeans = KMeans(n_clusters=4).fit(points) #총 클러스터 4개를 생성한다.
kmeans.cluster_centers_ #결과로 4가지 중심값이 나온다.
#별다른 설정이 없으면 하나의 무작위값으로, 기본적으로 k-means++가 적용된다. 
```




    array([[ 7.        ,  3.        ],
           [66.28571429, 21.14285714],
           [22.125     , 31.625     ],
           [58.        , 88.        ]])




```python
kmeans.labels_
```




    array([0, 0, 2, 1, 2, 3, 1, 1, 1, 3, 2, 2, 1, 2, 1, 2, 1, 3, 3, 2, 2])




```python
df['cluster'] = kmeans.labels_ 
#하나의 클러스터라는 속성을 만들어주고 그 값으로는 클러스터의 id값이 들어가도록 한다.
df.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
      <th>cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>24</td>
      <td>33</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>52</td>
      <td>13</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22</td>
      <td>30</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>62</td>
      <td>83</td>
      <td>3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>82</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>72</td>
      <td>38</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>92</td>
      <td>30</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>82</td>
      <td>93</td>
      <td>3</td>
    </tr>
    <tr>
      <th>10</th>
      <td>28</td>
      <td>33</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>23</td>
      <td>23</td>
      <td>2</td>
    </tr>
    <tr>
      <th>12</th>
      <td>62</td>
      <td>38</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>22</td>
      <td>23</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14</th>
      <td>52</td>
      <td>13</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>24</td>
      <td>37</td>
      <td>2</td>
    </tr>
    <tr>
      <th>16</th>
      <td>52</td>
      <td>13</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>26</td>
      <td>93</td>
      <td>3</td>
    </tr>
    <tr>
      <th>18</th>
      <td>62</td>
      <td>83</td>
      <td>3</td>
    </tr>
    <tr>
      <th>19</th>
      <td>12</td>
      <td>35</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
sb.lmplot('x','y',data=df, fit_reg=False, scatter_kws={"s": 150}, hue="cluster")
plt.title('k-means example')
```




    Text(0.5,1,'k-means example')




![png](output_9_1.png)

