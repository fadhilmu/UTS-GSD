# UTS Graf Sains Data (GSD)

| | |
|---|---|
| **Nama** | Fadhil Muhamad |
| **NIM** | 23031554090 |
| **Mata Kuliah** | Graf Sains Data |
| **Dataset** | RetailRocket — `events.csv` |
| **Tools** | Python (pandas, NumPy, SciPy, scikit-learn, NetworkX, Matplotlib), Neo4j + Python Driver |

---

## Daftar Isi

1. [Penentuan Periode Waktu](#1-penentuan-periode-waktu)
2. [Filter Timestamp dan Penyimpanan CSV](#2-filter-timestamp-dan-penyimpanan-csv)
3. [Komposisi Subgraph di Neo4j](#3-komposisi-subgraph-di-neo4j)
4. [Vector Space Modelling — User](#4-vector-space-modelling--user)
5. [Vector Space Modelling — Item](#5-vector-space-modelling--item)
6. [Perbandingan Proyeksi User–User vs Item–Item](#6-perbandingan-proyeksi-useruser-vs-itemitem)
7. [Perhitungan Similarity pada Proyeksi Item–Item](#7-perhitungan-similarity-pada-proyeksi-itemitem)
8. [Catatan Tambahan](#8-catatan-tambahan)

### Struktur File

| File | Isi |
|---|---|
| `data_event.ipynb` | Eksplorasi data mentah, penentuan periode, filter timestamp, ekspor CSV |
| `UTS-GSD_Fadhil090-push.ipynb` | Impor ke Neo4j, query komposisi, VSM, proyeksi, benchmark, similarity |
| `events_P1-P2.csv` | Data hasil filter (P1 + P2) dengan kolom `fold` |

---

## 1. Penentuan Periode Waktu

### Periode yang dipilih

| Fold | Periode | Durasi |
|---|---|---|
| **P1 — Training** | 01/06/2015 – 30/06/2015 | 30 hari |
| **P2 — Testing** | 01/07/2015 – 14/07/2015 | 14 hari |

### Dasar pemilihan

Eksplorasi awal pada `events.csv`:

- Total **2.756.101** interaksi, dengan rentang **03/05/2015 – 18/09/2015**.
- Distribusi event: `view` 2.664.312, `addtocart` 69.332, `transaction` 22.457.
- Rata-rata **±19.828 interaksi per hari**.

Alasan pemilihan periode:

1. **Menghindari tepi data.** Hari di awal dan akhir rentang tidak terekam penuh, misalnya 18/09 hanya 1.528 interaksi dan 03/05 hanya 13.683. Kedua periode berada di tengah rentang yang stabil.
2. **Pemisahan temporal tanpa jeda.** P2 langsung menyusul P1. Model belajar dari masa lalu (Juni) dan diuji pada perilaku dua minggu berikutnya. Skema ini meniru kondisi nyata sistem rekomendasi dan mencegah *data leakage* dari masa depan.
3. **Proporsi seimbang.** Pembagiannya sekitar **66% : 34%** (610.393 : 309.637 baris), mendekati pembagian umum train/test.

### Ringkasan hasil filter

| | P1 (Training) | P2 (Testing) |
|---|---:|---:|
| Interaksi | 610.393 | 309.637 |
| Visitor unik | 313.832 | 171.802 |
| Item unik | 117.317 | 83.507 |
| ID transaksi unik | 4.037 | 1.914 |
| `view` | 590.255 | 299.629 |
| `addtocart` | 15.095 | 7.535 |
| `transaction` | 5.043 | 2.473 |

> Jumlah event `transaction` lebih besar dari jumlah ID transaksi unik karena satu transaksi dapat berisi beberapa item. Setiap item tercatat sebagai satu baris event.

---

## 2. Filter Timestamp dan Penyimpanan CSV

```python
import pandas as pd
from pathlib import Path

df = pd.read_csv("events.csv")

# Timestamp RetailRocket berupa Unix epoch dalam milidetik
dt = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")

# Interval setengah terbuka [start, end)
mask_p1 = (dt >= "2015-06-01") & (dt < "2015-07-01")
mask_p2 = (dt >= "2015-07-01") & (dt < "2015-07-15")

df_p1 = df.loc[mask_p1].copy()
df_p1["fold"] = "P1_training"

df_p2 = df.loc[mask_p2].copy()
df_p2["fold"] = "P2_testing"

df_filtered = pd.concat([df_p1, df_p2], ignore_index=True)
df_filtered.to_csv(Path("events_P1-P2.csv"), index=False)
```

### Penjelasan

- **Konversi waktu.** `pd.to_datetime(..., unit="ms")` mengubah epoch milidetik menjadi tanggal. Kolom `timestamp` asli tetap disimpan agar bisa diimpor ke Neo4j sebagai properti relasi.
- **Batas periode.** Filter `>= start` dan `< end` memastikan seluruh interaksi 30 Juni sampai pukul 23:59:59 ikut terambil, dan tidak ada baris yang masuk ke dua fold sekaligus.
- **Label fold.** Kolom `fold` menandai asal periode, sehingga saat impor ke Neo4j data testing bisa dibuang.

### Verifikasi

| Fold | Jumlah | Mulai | Akhir |
|---|---:|---|---|
| `P1_training` | 610.393 | 2015-06-01 00:00:01 | 2015-06-30 23:59:59 |
| `P2_testing` | 309.637 | 2015-07-01 00:00:06 | 2015-07-14 23:59:56 |
| **Total** | **920.030** | | |

Kolom `fold` tidak memiliki nilai kosong. Nilai kosong hanya ada pada `transactionid` (912.514 baris), dan ini wajar karena kolom tersebut hanya terisi untuk event `transaction`.

---

## 3. Komposisi Subgraph di Neo4j

### Model graf

```
(:Visitor {visitorId}) -[:VIEWED        {timestamp, fold}]->                (:Item {itemId})
(:Visitor {visitorId}) -[:ADDED_TO_CART {timestamp, fold}]->                (:Item {itemId})
(:Visitor {visitorId}) -[:BOUGHT        {timestamp, transactionId, fold}]-> (:Item {itemId})
```

Graf bersifat **bipartit dan heterogen**. Dua jenis simpul (`Visitor`, `Item`) dihubungkan oleh tiga jenis relasi berarah. Hanya data **P1_training** yang diimpor ke graf.

### Hasil komposisi

| Jenis simpul | Jumlah |
|---|---:|
| `Visitor` | 313.832 |
| `Item` | 117.317 |
| **Total simpul** | **431.149** |

| Jenis relasi | Jumlah relasi | Visitor unik | Item unik | Proporsi |
|---|---:|---:|---:|---:|
| `VIEWED` | 590.255 | 313.218 | 117.153 | 96,7% |
| `ADDED_TO_CART` | 15.095 | 8.595 | 8.144 | 2,5% |
| `BOUGHT` | 5.043 | 2.728 | 3.589 | 0,8% |
| **Total** | **610.393** | | | |

**Interpretasi:**

- Total relasi (590.255 + 15.095 + 5.043 = 610.393) sama persis dengan jumlah baris P1.
- Query "relasi tanpa fold" mengembalikan **0**, dan jumlah simpul yang memiliki relasi apa pun sama dengan jumlah simpul P1. Artinya graf hanya berisi data training dan tidak ada kebocoran dari P2.
- Relasi `VIEWED` mendominasi. Aktivitas melihat produk jauh lebih sering daripada menambah ke keranjang atau membeli.
- Jumlah simpul unik per jenis relasi tidak bisa dijumlahkan, karena satu visitor atau item dapat terlibat di beberapa jenis interaksi.

### Cypher yang digunakan

**a. Constraint (keunikan + index)**

```cypher
CREATE CONSTRAINT visitor_id_unique IF NOT EXISTS
FOR (v:Visitor) REQUIRE v.visitorId IS UNIQUE;

CREATE CONSTRAINT item_id_unique IF NOT EXISTS
FOR (i:Item) REQUIRE i.itemId IS UNIQUE;
```

Constraint menjamin tidak ada simpul ganda. Neo4j juga otomatis membuat index, sehingga operasi `MERGE` saat impor menjadi cepat.

**b. Impor batch**

```cypher
UNWIND $batch AS row
MERGE (v:Visitor {visitorId: row.visitorid})
MERGE (i:Item    {itemId:    row.itemid})
WITH v, i, row
CALL (v, i, row) {
    WITH row WHERE row.event = 'view'
    CREATE (v)-[:VIEWED {timestamp: row.timestamp, fold: row.fold}]->(i)
    UNION ALL
    WITH row WHERE row.event = 'addtocart'
    CREATE (v)-[:ADDED_TO_CART {timestamp: row.timestamp, fold: row.fold}]->(i)
    UNION ALL
    WITH row WHERE row.event = 'transaction'
    CREATE (v)-[:BOUGHT {timestamp: row.timestamp,
                         transactionId: row.transactionid,
                         fold: row.fold}]->(i)
}
```

| Klausa | Fungsi |
|---|---|
| `UNWIND $batch AS row` | Memecah list record (5.000 baris per batch) dari Python menjadi baris-baris Cypher |
| `MERGE` (simpul) | Membuat simpul hanya jika belum ada, sehingga tidak ada duplikasi |
| `CALL (...) { ... UNION ALL ... }` | Memilih jenis relasi sesuai nilai kolom `event` |
| `CREATE` (relasi) | Setiap event dicatat sebagai relasi terpisah beserta `timestamp`-nya |

**c. Query komposisi**

```cypher
-- Jenis simpul dan jumlahnya
MATCH (n)
RETURN labels(n) AS jenis_simpul, count(n) AS jumlah
ORDER BY jumlah DESC;

-- Komposisi relasi
MATCH (v:Visitor)-[r]->(i:Item)
WHERE r.fold = 'P1_training'
RETURN type(r)            AS relationship_type,
       count(r)           AS total_relationships,
       count(DISTINCT v)  AS unique_visitors,
       count(DISTINCT i)  AS unique_items
ORDER BY relationship_type;

-- Pemeriksaan relasi tanpa fold
MATCH ()-[r]->() WHERE r.fold IS NULL
RETURN count(r);
```

| Klausa | Fungsi |
|---|---|
| `MATCH (v:Visitor)-[r]->(i:Item)` | Mencari pola Visitor → Item untuk relasi jenis apa pun |
| `WHERE r.fold = 'P1_training'` | Memfilter relasi berdasarkan properti |
| `labels(n)` / `type(r)` | Mengambil label simpul / nama jenis relasi |
| `count(r)` vs `count(DISTINCT v)` | Jumlah relasi vs jumlah simpul unik |
| Agregasi | Dikelompokkan otomatis berdasarkan kolom non-agregat, mirip `GROUP BY` di SQL |

**d. Python API**

```python
with driver.session(database=DATABASE) as session:
    result = session.run(query)
    composition = pd.DataFrame([dict(record) for record in result])
```

`driver.session()` membuka sesi ke database, `session.run()` mengeksekusi Cypher, dan setiap `Record` diubah menjadi `dict` lalu `DataFrame`.

---

## 4. Vector Space Modelling — User

Setiap visitor direpresentasikan sebagai **vektor biner** sepanjang jumlah item. Nilai `1` berarti visitor pernah berinteraksi dengan item tersebut (event apa pun, setelah `drop_duplicates`). Matriks disimpan dalam format sparse `csr_matrix`.

```python
df_p1 = df.loc[df["fold"].eq("P1_training"), ["visitorid", "itemid"]].drop_duplicates()
user_ids, user_idx = np.unique(df_p1["visitorid"], return_inverse=True)
item_ids, item_idx = np.unique(df_p1["itemid"],    return_inverse=True)

X_ui = csr_matrix(
    (np.ones(len(df_p1), dtype=np.float32), (user_idx, item_idx)),
    shape=(len(user_ids), len(item_ids))
)
```

### Karakteristik

| Karakteristik | Nilai |
|---|---:|
| Shape | 313.832 × 117.317 |
| Dimensi vektor | 117.317 (satu dimensi per item) |
| Nilai non-zero | 471.258 |
| Sparsity | 99,998720% |
| Rata-rata interaksi unik per user | 1,50 |
| Median | 1 |
| Maksimum | 1.031 |
| Memori (sparse) | 4,79 MB |

Validasi menunjukkan jumlah visitor dan item **sama dengan hasil Neo4j**, dan seluruh nilai matriks bernilai biner.

### Interpretasi

- **Sebagian besar user hanya menyentuh satu item** (median = 1). Vektornya hampir kosong, sehingga cosine similarity antar-user sering bernilai 0 karena tidak ada item yang sama.
- **Plot NetworkX.** Pada graf bipartit sampel (30 user × 40 item), proyeksi **User–User menghasilkan 0 edge** dan 30 komponen terpisah. Tidak ada pasangan user yang berbagi item.
- **Histogram distribusi** (skala log) memperlihatkan pola *long tail*: sangat banyak user dengan 1–2 interaksi, dan sedikit sekali user yang sangat aktif.

---

## 5. Vector Space Modelling — Item

Representasi item adalah **transpose** matriks user (`X_iu = X_ui.T`). Setiap item menjadi vektor biner sepanjang jumlah visitor.

### Karakteristik

| Karakteristik | Nilai |
|---|---:|
| Shape | 117.317 × 313.832 |
| Dimensi vektor | 313.832 (satu dimensi per visitor) |
| Nilai non-zero | 471.258 |
| Sparsity | 99,998720% |
| Rata-rata visitor unik per item | 4,02 |
| Median | 2 |
| Maksimum | 1.022 |

### Interpretasi

- Sparsity keseluruhan sama karena matriksnya sama. Namun **informasi per vektor lebih padat**: rata-rata 4,02 visitor per item dibanding 1,50 item per user.
- **Plot NetworkX.** Pada sampel yang sama, proyeksi **Item–Item menghasilkan 26 edge**, dengan density 0,0333 dan 22 komponen. Item saling terhubung melalui visitor yang sama.

---

## 6. Perbandingan Proyeksi User–User vs Item–Item

**Kesimpulan: proyeksi Item–Item lebih efektif sekaligus lebih efisien.**

| Aspek | User–User | Item–Item |
|---|---:|---:|
| Jumlah objek | 313.832 | **117.317** (2,68× lebih sedikit) |
| Pasangan similarity penuh (n²) | ≈ 9,85 × 10¹⁰ | **≈ 1,38 × 10¹⁰** (≈7× lebih kecil) |
| Rata-rata interaksi per vektor | 1,50 | **4,02** |
| Median interaksi | 1 | **2** |
| Edge pada proyeksi sampel | 0 | **26** |
| Waktu query kNN (20 anchor) | 0,515 detik | **0,189 detik** |
| Waktu query per anchor | 25,77 ms | **9,46 ms** (≈63% lebih cepat) |
| Objek P2 yang sudah dikenal di P1 | ≈5,8% visitor | **≈66,7% item** |

### Efisiensi

- Objek yang perlu dibandingkan lebih sedikit. Biaya similarity tumbuh **kuadratik** terhadap jumlah objek, sehingga selisih 2,68× pada jumlah objek menjadi sekitar 7× pada jumlah pasangan.
- Benchmark kNN (cosine, `brute`, k = 5) menunjukkan query Item–Item sekitar 2,7× lebih cepat. Waktu *fit* Item–Item sedikit lebih lama (0,047 vs 0,014 detik), tetapi pada algoritma `brute` tahap ini hanya menyimpan matriks sehingga selisihnya tidak berarti.

### Efektivitas

- **Vektor item lebih kaya informasi.** Peluang dua vektor memiliki irisan jauh lebih besar, sehingga skor similarity bermakna. Pada vektor user yang mayoritas hanya berisi satu interaksi, similarity hampir selalu 0.
- **Cakupan pada periode testing.**

  | | Muncul di P2 | Baru di P2 | Sudah dikenal dari P1 |
  |---|---:|---:|---:|
  | Visitor | 171.802 | 161.845 | **9.957 (≈5,8%)** |
  | Item | 83.507 | 27.794 | **55.713 (≈66,7%)** |

  Sekitar **94% visitor di P2 adalah user baru** yang tidak memiliki vektor di P1, sehingga pendekatan User–User tidak dapat melayani mereka. Sebaliknya, sekitar dua pertiga item di P2 sudah memiliki vektor dari P1.
- Profil item juga cenderung lebih stabil dari waktu ke waktu dibanding perilaku user, sehingga cocok dengan skema evaluasi temporal P1 → P2.

> Batasan: benchmark waktu dilakukan pada 20 anchor dan dipengaruhi oleh lingkungan komputasi.

---

## 7. Perhitungan Similarity pada Proyeksi Item–Item

### Metode

- **Item acuan (anchor):** item dengan visitor terbanyak, yaitu **Item `5411`** dengan **1.022 visitor unik**.
- **Metrik:** cosine similarity antara vektor anchor dan seluruh vektor item.
- **Efisiensi memori:** hanya dihitung **satu baris** similarity (1 × 117.317), bukan matriks penuh 117.317 × 117.317.
- Anchor diberi skor −1 agar tidak muncul sebagai item mirip dengan dirinya sendiri.

```python
anchor_idx    = int(np.argmax(np.diff(X_iu.indptr)))
anchor_vector = X_iu[anchor_idx]

similarity_scores = cosine_similarity(anchor_vector, X_iu).ravel()
similarity_scores[anchor_idx] = -1
top_indices = np.argsort(similarity_scores)[-5:][::-1]
```

Untuk vektor biner, cosine similarity dapat disederhanakan menjadi:

$$
\cos(A, B) = \frac{|A \cap B|}{\sqrt{|A| \cdot |B|}}
$$

dengan $|A \cap B|$ adalah jumlah visitor yang berinteraksi dengan kedua item. Nilai mendekati 1 berarti pola visitor sangat mirip, sedangkan 0 berarti tidak ada visitor yang sama.

### Hasil Top-5 item paling mirip dengan Item `5411`

| Rank | Similar Item | Cosine Similarity | Shared Visitors | Visitor Item Tersebut |
|---:|---:|---:|---:|---:|
| 1 | 33030 | 0,181815 | 50 | 74 |
| 2 | 358234 | 0,102162 | 16 | 24 |
| 3 | 33956 | 0,100036 | 15 | 22 |
| 4 | 375492 | 0,074775 | 20 | 70 |
| 5 | 89143 | 0,064586 | 9 | 19 |

**Verifikasi manual (rank 1):**

$$
\frac{50}{\sqrt{1022 \times 74}} = \frac{50}{275{,}0} \approx 0{,}1818
$$

**Interpretasi:**

- Item `33030` paling mirip dengan `5411`. Sebanyak 50 dari 74 visitor-nya juga berinteraksi dengan `5411`.
- Item `375492` memiliki 20 shared visitors, lebih banyak dari rank 2 dan 3, tetapi berada di rank 4. Penyebabnya adalah normalisasi oleh ukuran vektor: item dengan banyak visitor (70) tidak otomatis dianggap paling mirip.

### Penyimpanan ke Neo4j

Hasil similarity disimpan kembali ke graf sebagai relasi `SIMILAR_TO`:

```cypher
UNWIND $rows AS row
MATCH (a:Item {itemId: row.`Anchor Item`})
MATCH (b:Item {itemId: row.`Similar Item`})
MERGE (a)-[r:SIMILAR_TO]->(b)
SET r.score                   = row.`Cosine Similarity`,
    r.sharedVisitors          = row.`Shared Visitors`,
    r.similarItemVisitorCount = row.`Similar Item Visitor Count`,
    r.fold                    = 'P1_training'
```

Verifikasi:

```cypher
MATCH (a:Item {itemId: 5411})-[r:SIMILAR_TO]->(b:Item)
RETURN a.itemId, b.itemId, r.score, r.sharedVisitors
ORDER BY r.score DESC;
```

Query mengembalikan lima relasi yang sama dengan hasil Python.

### Visualisasi sebelum dan sesudah similarity

| | Sebelum | Sesudah |
|---|---|---|
| Query | Item `5411` beserta relasi interaksi dari visitor | Item `5411` beserta relasi `SIMILAR_TO` |
| Isi graf | Banyak simpul `Visitor` terhubung ke satu item | 6 simpul `Item` (1 anchor + 5 item mirip) dan 5 relasi |
| Makna | Pola interaksi pengguna terhadap item | Kemiripan antaritem berdasarkan visitor bersama |

Graf kini diperkaya dengan informasi kemiripan antaritem yang dapat ditelusuri langsung di Neo4j dan menjadi dasar rekomendasi berbasis item.

---

## 8. Catatan Tambahan

### Eksplorasi: evaluasi User-Based KNN

Sebagai pembanding, dilakukan evaluasi rekomendasi **User-Based KNN** pada satu user acak (`random_seed = 42`).

| | Nilai |
|---|---|
| User | `951502` |
| Item relevan di P2 (belum muncul di P1) | `264152` |
| Top-5 rekomendasi | `162411`, `371985`, `138523`, `457181`, `164027` |
| Hit@5 | 0 |
| NDCG@5 | 0,0 |

Tidak ada item rekomendasi yang cocok dengan item relevan. Hasil ini **konsisten dengan temuan soal 6**: vektor user sangat jarang, dan mayoritas user P2 tidak memiliki riwayat di P1, sehingga pendekatan User–User sulit menghasilkan rekomendasi yang tepat. Evaluasi hanya pada satu user, sehingga belum mewakili performa model secara keseluruhan.

### Lingkungan

| Library | Versi |
|---|---|
| Neo4j Python Driver | 6.4.0 |
| pandas | 2.2.3 |
| NetworkX | 3.4.2 |
| NumPy | 1.26.4 |
| scikit-learn | 1.7.0 |
| Python | 3.11.5 |
