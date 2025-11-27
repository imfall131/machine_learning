#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import numpy as np
df = pd.read_csv(r'C:\Users\gaeul kim\machine_fall\BostonHousing.csv')


# In[2]:


X = df.drop("medv", axis=1)  # 특성(feature)
y = df["medv"]  


# In[3]:


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 정규화된 데이터를 DataFrame으로 변환 (컬럼명 유지)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)


# In[4]:


df_scaled = pd.concat([X_scaled_df, y], axis=1)

# 정규화된 데이터 출력 (앞부분만)
print(df_scaled.head())


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)


# In[6]:


model = LinearRegression()
model.fit(X_train, y_train)

# 2. 학습 데이터에 대한 예측과 정확도
y_train_pred = model.predict(X_train)
train_score = r2_score(y_train, y_train_pred)

# 3. 테스트 데이터에 대한 예측과 정확도
y_test_pred = model.predict(X_test)
test_score = r2_score(y_test, y_test_pred)

# 4. 출력
print("학습 데이터 정확도 (R²):", train_score)
print("테스트 데이터 정확도 (R²):", test_score)


# In[7]:


alphas = [0.001, 0.01, 0.1, 1, 10, 100]

print("alpha\t학습 정확도(R²)\t테스트 정확도(R²)")
print("-" * 40)

# 각각의 alpha 값에 대해 Ridge 모델 학습 및 평가
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)

    # R² 계산
    train_score = ridge.score(X_train, y_train)
    test_score = ridge.score(X_test, y_test)

    print(f"{alpha:<7}\t{train_score:.4f}\t\t{test_score:.4f}")


# In[8]:


alphas = [0.001, 0.01, 0.1, 1, 10, 100]

print("alpha\t학습 정확도(R²)\t테스트 정확도(R²)")
print("-" * 40)

# 각각의 alpha 값에 대해 Ridge 모델 학습 및 평가
for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)

    # R² 계산
    train_score = lasso.score(X_train, y_train)
    test_score = lasso.score(X_test, y_test)

    print(f"{alpha:<7}\t{train_score:.4f}\t\t{test_score:.4f}")


# In[9]:


# Lasso 모델 생성 및 학습 (적절한 alpha 선택)
lasso = Lasso(alpha=0.1)  # alpha는 실험적으로 조절 가능
lasso.fit(X_train, y_train)

# 특성 이름 가져오기
feature_names = df.drop("medv", axis=1).columns

# 각 특성의 계수 절댓값과 함께 묶기
coef_abs = np.abs(lasso.coef_)
important_features = sorted(zip(feature_names, coef_abs), key=lambda x: x[1], reverse=True)

# 영향력이 큰 상위 3개 특성 출력
print("영향력이 큰 특성 Top 3:")
for name, coef in important_features[:3]:
    print(f"{name}: 계수 크기 = {coef:.4f}")

