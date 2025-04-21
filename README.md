# İşe Alım Tahmin API

Bu proje, başvuranların deneyim yılı ve teknik puanına göre işe alınıp alınmayacağını tahmin etmek için bir SVM modeli kullanan bir FastAPI uygulamasıdır.

## Kullanım

1.  Gerekli kütüphaneleri yükleyin
2.  API'yi çalıştırın: `python api.py`
3.  API belgelerine `http://127.0.0.1:8000/docs` adresinden erişin.
4.  `/predict` endpoint'ine POST isteği göndererek tahmin yapın. İstek gövdesi şu formatta olmalıdır:
    ```json
    {
      "experienceYears": 5,
      "technicalScore": 85
    }
    ```

## Dosyalar

*   `api.py`: FastAPI uygulamasının ana dosyası.
*   `bestModel.pkl`: Eğitilmiş SVM modeli.
*   `scaler.pkl`: Veri ölçekleyici.
*   `modelTuning.py`: Model hiperparametre optimizasyonu için kod.
*   `linearSVM.py`: SVM modelinin implementasyonu ve eğitimi.
*   `applicants.json`: Başvuru verileri.
*   `randomData.py`: Rastgele veri oluşturma betiği. 
