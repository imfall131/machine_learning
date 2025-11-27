import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split





def make_wave(n_samples=100):
    rnd=np.random.RandomState(42)
    x=rnd.uniform(-3,3,size=n_samples)
    y_no_noise= (np.sin(4*x)+x)
    y=(y_no_noise + rnd.normal(size=len(x)))/2
    return x.reshape(-1,1), y





x, y = make_wave()
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
x_range = np.linspace(-3, 3, 1000).reshape(-1, 1)




k_values = [1, 3, 5, 7, 10]





for k in k_values:
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(x_train, y_train)
    
    y_pred_curve = model.predict(x_range)
    y_test_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_test_pred)
    
    # 각 그래프를 개별로 출력
    plt.figure(figsize=(7, 3))
    plt.scatter(x_train, y_train, color='skyblue', label='Train Data')
    plt.scatter(x_test, y_test, color='orange', label='Test Data')
    plt.plot(x_range, y_pred_curve, color='red', label='Prediction Curve')
    plt.title(f"k={k}, MSE={mse:.2f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()





#mse 차이가 적기 때문에 k값이 1,7,10 중에 하나를 택해야 한다. 하지만 k값이 1이라면 훈련데이터에 과접합될 가능성이 크고, k가 10이라면 너무 평탄한 그래프가 나올 수 있다. 






