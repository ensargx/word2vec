# Word2Vec Skip-Gram

Bu proje, **Word2Vec Skip-Gram** modelini kullanarak farklı embedding (kelime gömme) boyutlarının ve eğitim sürelerinin (epoch) anlamsal ilişkiler üzerindeki etkisini incelemektedir. Kelimeleri vektör uzayında temsil ederek aralarındaki anlamsal ve sözdizimsel ilişkiler matematiksel olarak analiz edilmiştir.

---

## 🛠 Model Mimarisi ve Matematiksel Temeller

Word2Vec, kelimeleri yalnızca sembol olarak değil, anlamsal ilişkileri koruyan yoğun vektörler (dense vectors) olarak temsil eder. Bu çalışmada kullanılan **Skip-Gram** modeli, bir merkez kelimeden ($w_t$) yola çıkarak belirli bir pencere boyutundaki bağlam kelimelerini ($w_{t+j}$) tahmin etmeyi amaçlar.

### 1. Skip-Gram Amaç Fonksiyonu
Modelin temel hedefi, veri kümesindeki tüm kelimeler için ortalama log-olasılığı maksimize etmektir:
$$\mathcal{L} = \frac{1}{T} \sum_{t=1}^{T} \sum_{-c \le j \le c, j \neq 0} \log p(w_{t+j} | w_t)$$
Burada $c$ bağlam penceresinin boyutunu, $T$ ise toplam kelime sayısını ifade eder.

### 2. Negative Sampling ve Hesaplama Verimliliği
Bir bağlam kelimesinin olasılığı normal şartlarda **Softmax** fonksiyonu ile hesaplanır:
$$p(w_O | w_I) = \frac{\exp({v'_{w_O}}^\top v_{w_I})}{\sum_{w=1}^{V} \exp({v'_{w}}^\top v_{w_I})}$$
Ancak sözlük boyutu ($V$) on binlerce kelimeye ulaştığında, paydadaki toplam işlemi hesaplama açısından çok maliyetlidir. Bu sorunu aşmak için projemizde **Negative Sampling** yöntemi uygulanmıştır:

$$E = \log \sigma({v'_{w_O}}^\top v_{w_I}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)} \left[ \log \sigma(-{v'_{w_i}}^\top v_{w_I}) \right]$$

Bu yöntemde, her adımda tüm sözlüğü güncellemek yerine bir pozitif örnek ve $k$ adet negatif örnek seçilerek model eğitilir. Negatif kelimeler seçilirken frekans dağılımının **3/4** kuvveti ($P_n(w) = U(w)^{3/4}/Z$) kullanılarak nadir kelimelerin de örneklenmesi sağlanmıştır.

### 3. Subsampling (Alt Örnekleme)
"the", "of", "and" gibi çok sık geçen ve anlamsal değeri düşük kelimelerin etkisini azaltmak için şu olasılık formülü ile eğitimden elenmesi sağlanmıştır:
$$P(w_i) = 1 - \sqrt{\frac{t}{f(w_i)}}$$
Burada $f(w_i)$ kelimenin frekansını, $t$ ise eşik değerini temsil eder.

---

## 📊 Veri Kümesi ve Ön İşleme

*   **Kaynak:** [afmck/text8](https://huggingface.co/datasets/afmck/text8) (Temizlenmiş Wikipedia verisi).
*   **İşlem:** 5 kereden az geçen kelimeler sözlükten çıkarılmıştır.
*   **Kapsam:** 5,567,194 kelimelik eğitim verisi ve 71,290 kelimelik bir sözlük yapısı oluşturulmuştur.

---

## 🧪 Analoji Performans Karşılaştırması

Analoji testleri ($a - b + c = ?$), modelin öğrendiği vektörlerin uzaydaki tutarlılığını ölçmek için kullanılmıştır. Aşağıdaki tablo, 15 Epoch eğitim sonrası elde edilen sıralama (**Rank**) değerlerini göstermektedir.

| Kategori | Analoji Sorusu | Beklenen | 128-dim | 256-dim | 512-dim |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Aile** | `king - man + woman` | **queen** | 23 | 5 | **2** |
| **Aile** | `uncle - man + woman` | **aunt** | 4 | **1** | **1** |
| **Sıfat** | `bigger - big + smaller` | **small** | **3** | 6 | 48 |
| **Sıfat** | `harder - hard + easier` | **easy** | **11** | 101 | 236 |
| **Fiil** | `sang - sing + danced` | **dance** | 87 | **7** | 9 |
| **Başkent**| `japan - tokyo + egypt` | **cairo** | 386 | 479 | **250** |

---

## 📈 Önemli Bulgular

*   **Vektör Boyutu ve Anlam:** Aile ve cinsiyet gibi kategorilerde 512 boyutlu vektörler, anlamsal ilişkileri **Rank 1** (tam isabet) seviyesine kadar optimize edebilmiştir.
*   **Sıfat Paradoksu:** Şaşırtıcı bir şekilde, 128 boyutlu model sıfat analojilerinde daha yüksek sıralama başarısı göstermiştir. Bu, düşük boyutların daha kısıtlı ama keskin bir genelleme yapabildiğini göstermektedir.
*   **Eğitim Derinliği:** 15 epoch, 10 epoch'a kıyasla neredeyse her parametrede kosinüs benzerliği skorlarını stabilize etmiştir.
