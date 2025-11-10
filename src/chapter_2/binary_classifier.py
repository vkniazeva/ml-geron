from src.chapter_2.data_loading import *
from sklearn.linear_model import SGDClassifier

X_train, X_test, y_train, y_test = load_data(60000)

Y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

sgd_classifier = SGDClassifier(random_state=42)
sgd_classifier.fit(X_train, Y_train_5)
digit_vector, digit_image = convert_to_image(X_train.to_numpy(), 0)
print(sgd_classifier.predict([digit_vector]))





