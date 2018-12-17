from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
import pandas as pd

dataset = pd.read_csv("data.csv",encoding="gbk")
x_train,x_test,y_train,y_test = train_test_split(dataset.values[:,:-1],dataset.values[:,-1],test_size=0.3,random_state=10)

def default():
    model = RandomForestClassifier()
    model.fit(x_train,y_train)
    print(model.score(x_test,y_test))
#树的颗数 深度
def adjust_n_estimators():
    params={"n_estimators":range(15,25),"max_depth":[11,13,15,17,19]}
    gs = GridSearchCV(RandomForestClassifier(),param_grid=params,scoring="accuracy")
    gs.fit(x_train,y_train)
    print(gs.best_score_)
    print(gs.best_params_)
#叶子节点数
#叶子切分点数
#画出roc曲线
#最佳的参数
def best():
    model = RandomForestClassifier(n_estimators=21,max_depth=11)
    model.fit(x_train,y_train)
    print(model.score(x_train,y_train))
    print(model.score(x_test,y_test))
if __name__=="__main__":
    # default()
    # adjust_n_estimators()
    best()