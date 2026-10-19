# Farklı uzunluklu prompt'lar: static batch

WSL, RTX 4070 Laptop, Qwen2.5-0.5B-Instruct fp16. Her dizi 128 token üretiyor (eos yok sayılıyor),
iki tarafa aynı token id'leri. Throughput uçtan uca (prefill dahil). TTFT: bütün dizilerin ilk
token'ı, TPOT: adım aralıklarının medyanı.

## vLLM ile (4457b6f, paketli; 3 tekrarın medyanı)

| Durum | Uzunluklar (toplam) | Bizim tok/s | vLLM tok/s | Bizim TTFT | vLLM TTFT | Bizim TPOT | vLLM TPOT |
|---|---|---|---|---|---|---|---|
| B=2 karışık | 1800, 200 (2000) | 373 | 251 | 57.0 ms | 70.8 ms | 4.85 ms | 7.13 ms |
| B=4 karışık | 2000, 1100, 400, 90 (3590) | 691 | 491 | 97.1 ms | 121.5 ms | 4.99 ms | 7.12 ms |
| B=8 karışık | 1900, 1500, 1200, 800, 500, 300, 150, 60 (6410) | 1188 | 859 | 173.2 ms | 219.5 ms | 5.24 ms | 7.45 ms |
| B=8 1 uzun | 2000 + 7 x 64 (2448) | 1427 | 1010 | 71.4 ms | 89.1 ms | 5.00 ms | 7.11 ms |
| B=8 rastgele | 1432, 109, 1727, 1394, 619, 1188, 1778, 1342 (9589) | 1072 | 791 | 260.1 ms | 313.0 ms | 5.40 ms | 7.62 ms |

## Pad'li (7d0e916) ve paketli (4457b6f) (5 tekrarın medyanı)

Pad'li: B x T_max, right padding, derlenmiş prefill, sdpa. Paketli: (1, total_T), Triton varlen prefill.

| Durum | Pad'li tok/s | Paketli tok/s | Pad'li TTFT | Paketli TTFT | Pad'li TPOT | Paketli TPOT |
|---|---|---|---|---|---|---|
| B=2 karışık | 349 | 371 | 99.4 ms | 57.1 ms | 4.91 ms | 4.89 ms |
| B=4 karışık | 590 | 688 | 219.6 ms | 99.6 ms | 4.97 ms | 4.99 ms |
| B=8 karışık | 935 | 1236 | 417.8 ms | 170.7 ms | 5.24 ms | 5.17 ms |
| B=8 1 uzun | 938 | 1459 | 441.7 ms | 69.0 ms | 5.02 ms | 4.95 ms |
| B=8 rastgele | 938 | 1090 | 395.2 ms | 260.5 ms | 5.40 ms | 5.33 ms |
