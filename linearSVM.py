import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_json('applicants.json', orient='records')

X = df[['experienceYears', 'technicalScore']]
y = df['hiring']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svc = SVC(kernel='linear', random_state=42, probability=True)
svc.fit(X_train_scaled, y_train)
y_pred = svc.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
confusionMatrix = confusion_matrix(y_test, y_pred)
classReport = classification_report(y_test, y_pred, target_names=['Hired (0)', 'Not Hired (1)'])

print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(confusionMatrix)
print("Classification Report:")
print(classReport)

plt.figure(figsize=(8, 6))
X_min, X_max = X_train['experienceYears'].min() - 1, X_train['experienceYears'].max() + 1
y_min, y_max = X_train['technicalScore'].min() - 1, X_train['technicalScore'].max() + 1
xx, yy = np.meshgrid(np.arange(X_min, X_max, 0.02), np.arange(y_min, y_max, 0.02))
grid_points_scaled = scaler.transform(np.c_[xx.ravel(), yy.ravel()])
Z = svc.predict(grid_points_scaled)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.6, cmap=plt.cm.coolwarm)
plt.scatter(X['experienceYears'], X['technicalScore'], c=y, cmap=plt.cm.coolwarm, edgecolors='k', s=50)
plt.title('SVM Karar Sınırı')
plt.xlabel('Deneyim Yılı')
plt.ylabel('Teknik Skor')
plt.show()

print("\n--- Tahmin Yapalım ---")
while True:
    try:
        experienceInput = input("Deneyim yılınızı giriniz (çıkmak için q'ya basınız) ")
        if experienceInput.lower() == 'q':
            break
        experienceInput = float(experienceInput)

        scoreInput = input("Teknik puanınızı giriniz (çıkmak için q'ya basınız) ")
        if scoreInput.lower() == 'q':
            break
        scoreInput = float(scoreInput)

        if not (0 <= experienceInput <= 10):
             print("Deneyim yılı 1-10 arasında olmalıdır")
             continue
        if not (0 <= scoreInput <= 100):
             print("Teknik puan 0-100 arasında olmalıdır")
             continue
        userInputDf = pd.DataFrame([[experienceInput, scoreInput]], columns=['experienceYears', 'technicalScore'])
        userInputScaled = scaler.transform(userInputDf)
        prediction = svc.predict(userInputScaled)
        probabilities = svc.predict_proba(userInputScaled)
        resultText = "Red" if prediction[0] == 1 else "Kabul"
        confidenceNotHired = probabilities[0][1] * 100
        confidenceHired = probabilities[0][0] * 100

        print(f"Sonuç: '{resultText}'")
        print(f"Red ihtimali (%): {confidenceNotHired:.2f}%")
        print(f"Kabul ihtimali (%): {confidenceHired:.2f}%")
        print("-" * 30)

    except ValueError:
        print("Geçersiz giriş yaptınız, lütfen sayı giriniz.")
    except Exception as e:
        print(f"Hata: {e}")

print("Programdan çıkılıyor...")