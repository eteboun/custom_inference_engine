# kernels/decode_attention.py'deki GQA decode attention kernel'i için parametre taraması:
# L x BLOCK_N x num_warps x P, P > ceil(L / BLOCK_N) olanlar elenir.
# her konfigürasyon ve cuDNN SDPA için warm/cold latency + doğruluk ölçülür,
# sonuçlar results/gqa_sweep.html'e yazılır. proje kökünden: python -m benchmarks.gqa_sweep
#
# ölçüm (benchmarks/decode_attn.py ile aynı warm/cold yöntemi):
# cold: n_sets farklı (q, k, v), toplamı L2'nin birkaç katı, graph hepsini random
# sırayla çalıştırır. warm: aynı sayıda çağrı ama hep ilk set.
# bütün graph'lar (bütün L'ler, triton + cudnn, warm + cold) tek bir random sırada
# önce WARMUP_ITERS kez çalıştırılır, sonra her biri ITERS kez ölçülür (ölçüm sırası
# yine random). sonuç: çağrı başına sürenin ortalaması (± std).
import datetime
import html
import itertools
import random
import statistics
import time

import torch
import torch.nn.functional as F
import triton
from torch.nn.attention import sdpa_kernel, SDPBackend

from kernels.decode_attention import decode_gqa_attention_kernel_load, decode_attention_kernel_merge
from benchmarks.bench_utils import capture_graph, COLD_FOOTPRINT_L2_MULTIPLE

SEQ_LENS = [1024, 4096, 8192, 16384, 32768]
BLOCK_NS = [16, 32, 64, 128]
NUM_WARPS = [4, 8]
PS = [8, 16, 32, 64]

H_Q, H_KV, D = 14, 2, 64
GROUP_SIZE = H_Q // H_KV
BLOCK_G = max(16, triton.next_power_of_2(GROUP_SIZE))
BLOCK_D = triton.next_power_of_2(D)

WARMUP_ITERS = 3
ITERS = 10
SEED = 0
HTML_PATH = "results/gqa_sweep.html"


def make_data(L):
    l2_bytes = torch.cuda.get_device_properties(0).L2_cache_size
    set_bytes = (2 * H_KV * L + H_Q) * D * 2
    n_sets = triton.cdiv(COLD_FOOTPRINT_L2_MULTIPLE * l2_bytes, set_bytes)
    return {
        "n_sets": n_sets,
        "qs": torch.randn((n_sets, H_Q, D), device="cuda", dtype=torch.float16),
        "ks": torch.randn((n_sets, H_KV, L, D), device="cuda", dtype=torch.float16),
        "vs": torch.randn((n_sets, H_KV, L, D), device="cuda", dtype=torch.float16),
    }


def reference(data):
    # fp32, her q head kendi kv head'ini kullanır
    q = data["qs"][0].float()
    k = data["ks"][0].float().repeat_interleave(GROUP_SIZE, dim=0)
    v = data["vs"][0].float().repeat_interleave(GROUP_SIZE, dim=0)
    scores = torch.einsum("hld,hd->hl", k, q) * D ** -0.5
    return torch.einsum("hl,hld->hd", torch.softmax(scores, dim=-1), v)


def make_triton(L, data, BLOCK_N, num_warps, P):
    # decode_gqa_attention ile aynı iki kernel, ama ara buffer'lar bir kez ayrılıyor
    # (graph'taki bütün çağrılar sırayla aynı buffer'ları kullanır) ve num_warps verilebiliyor
    qs, ks, vs = data["qs"], data["ks"], data["vs"]
    partial_m = torch.empty((H_Q, P), device="cuda", dtype=torch.float32)
    partial_l = torch.empty((H_Q, P), device="cuda", dtype=torch.float32)
    partial_acc = torch.empty((H_Q, P, D), device="cuda", dtype=torch.float32)
    o = torch.empty((H_Q, D), device="cuda", dtype=torch.float16)
    seq_len = torch.tensor(L, device="cuda", dtype=torch.int32)

    def run(i):
        # kernel batch'li, burada B = 1 (grid'in 3. boyutu 1, batch adımı kullanılmıyor)
        decode_gqa_attention_kernel_load[(P, H_KV, 1)](
            qs[i], ks[i], vs[i],
            seq_len,
            partial_m, partial_l, partial_acc,
            ks.stride(0), ks.stride(1),
            D=D, Q_HEADS=H_Q, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            BLOCK_G=BLOCK_G, GROUP_SIZE=GROUP_SIZE,
            num_warps=num_warps,
        )
        decode_attention_kernel_merge[(H_Q, 1)](
            o, partial_m, partial_l, partial_acc,
            P=P, D=D, Q_HEADS=H_Q, BLOCK_P=triton.next_power_of_2(P), BLOCK_D=BLOCK_D,
        )
        return o

    return run


def make_cudnn(L, data):
    qs, ks, vs = data["qs"], data["ks"], data["vs"]

    def run(i):
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            return F.scaled_dot_product_attention(
                qs[i].view(1, H_Q, 1, D),
                ks[i].view(1, H_KV, L, D),
                vs[i].view(1, H_KV, L, D),
                enable_gqa=True,
            ).view(H_Q, D)

    return run


def check(out, ref, cudnn_out):
    err = (out.float() - ref).abs().max().item()
    try:
        torch.testing.assert_close(out, cudnn_out, rtol=1e-2, atol=1e-2)
        ok = True
    except AssertionError:
        ok = False
    return err, ok


def time_replay(graph, n_calls):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / n_calls  # us / çağrı


def main():
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    all_configs = list(itertools.product(SEQ_LENS, BLOCK_NS, NUM_WARPS, PS))
    configs = [c for c in all_configs if c[3] <= triton.cdiv(c[0], c[1])]
    skipped = [c for c in all_configs if c not in configs]
    print(f"{len(all_configs)} konfigürasyon, {len(skipped)} elendi "
          f"(P > ceil(L / BLOCK_N)), {len(configs)} ölçülecek")

    datas, refs, cudnn_outs = {}, {}, {}
    graphs, n_calls, meta = {}, {}, {}

    for L in SEQ_LENS:
        data = datas[L] = make_data(L)
        n = data["n_sets"]
        refs[L] = reference(data)
        cold = list(range(n))
        rng.shuffle(cold)
        orders = {"warm": [0] * n, "cold": cold}

        cudnn = make_cudnn(L, data)
        cudnn_outs[L] = cudnn(0).clone()
        meta[("cudnn", L)] = check(cudnn_outs[L], refs[L], cudnn_outs[L])
        for cache, order in orders.items():
            key = ("cudnn", L, cache)
            graphs[key] = capture_graph([lambda i=i: cudnn(i) for i in order])
            n_calls[key] = n

        for (cL, BN, W, P) in configs:
            if cL != L:
                continue
            run = make_triton(L, data, BN, W, P)
            cfg = ("triton", L, BN, W, P)
            meta[cfg] = check(run(0).clone(), refs[L], cudnn_outs[L])
            for cache, order in orders.items():
                graphs[cfg + (cache,)] = capture_graph([lambda i=i: run(i) for i in order])
                n_calls[cfg + (cache,)] = n
        print(f"L={L}: n_sets={n}, graph'lar hazır ({time.time() - t0:.0f} s)", flush=True)

    keys = list(graphs)
    warmup = keys * WARMUP_ITERS
    rng.shuffle(warmup)
    for key in warmup:
        graphs[key].replay()
    torch.cuda.synchronize()

    schedule = keys * ITERS
    rng.shuffle(schedule)
    samples = {key: [] for key in keys}
    for key in schedule:
        samples[key].append(time_replay(graphs[key], n_calls[key]))
    print(f"ölçüm bitti ({time.time() - t0:.0f} s, {len(schedule)} replay)")

    stats = {k: (statistics.mean(s), statistics.stdev(s)) for k, s in samples.items()}

    rows = []
    for (L, BN, W, P) in configs:
        cfg = ("triton", L, BN, W, P)
        err, ok = meta[cfg]
        rows.append({
            "L": L, "BLOCK_N": BN, "num_warps": W, "P": P,
            "warm": stats[cfg + ("warm",)], "cold": stats[cfg + ("cold",)],
            "cudnn_warm": stats[("cudnn", L, "warm")][0],
            "cudnn_cold": stats[("cudnn", L, "cold")][0],
            "err": err, "ok": ok,
        })

    cudnn = {L: {"warm": stats[("cudnn", L, "warm")], "cold": stats[("cudnn", L, "cold")],
                 "err": meta[("cudnn", L)][0]} for L in SEQ_LENS}
    write_html(rows, cudnn, skipped, time.time() - t0)

    for L in SEQ_LENS:
        r = [x for x in rows if x["L"] == L]
        bw = min(r, key=lambda x: x["warm"][0])
        bc = min(r, key=lambda x: x["cold"][0])
        print(f"L={L:>6}  en iyi warm {bw['warm'][0]:7.2f} us (BN={bw['BLOCK_N']}, "
              f"w={bw['num_warps']}, P={bw['P']}) cudnn {cudnn[L]['warm'][0]:7.2f} | "
              f"en iyi cold {bc['cold'][0]:7.2f} us (BN={bc['BLOCK_N']}, w={bc['num_warps']}, "
              f"P={bc['P']}) cudnn {cudnn[L]['cold'][0]:7.2f}")
    print(f"doğruluk: {sum(r['ok'] for r in rows)}/{len(rows)} geçti")
    print(f"html: {HTML_PATH}")


def kv_gbps(L, t_us):
    return 2 * H_KV * L * D * 2 / (t_us / 1e6) / 1e9


def write_html(rows, cudnn, skipped, elapsed):
    props = torch.cuda.get_device_properties(0)
    esc = html.escape

    def num(x, d=2):
        return f"{x:.{d}f}"

    summary = []
    for L in SEQ_LENS:
        r = [x for x in rows if x["L"] == L]
        bw = min(r, key=lambda x: x["warm"][0])
        bc = min(r, key=lambda x: x["cold"][0])
        cw, cc = cudnn[L]["warm"][0], cudnn[L]["cold"][0]
        summary.append(
            f"<tr><td>{L}</td>"
            f"<td>{num(bw['warm'][0])}</td><td class='cfg'>{bw['BLOCK_N']} / {bw['num_warps']} / {bw['P']}</td>"
            f"<td>{num(cw)}</td><td class='{ 'win' if cw / bw['warm'][0] >= 1 else 'lose'}'>{num(cw / bw['warm'][0])}x</td>"
            f"<td>{num(bc['cold'][0])}</td><td class='cfg'>{bc['BLOCK_N']} / {bc['num_warps']} / {bc['P']}</td>"
            f"<td>{num(cc)}</td><td class='{ 'win' if cc / bc['cold'][0] >= 1 else 'lose'}'>{num(cc / bc['cold'][0])}x</td>"
            f"<td>{num(kv_gbps(L, bc['cold'][0]), 0)}</td></tr>"
        )
        bw["best_warm"] = bc["best_cold"] = True

    body = []
    for x in rows:
        L = x["L"]
        (wm, ws), (cm, cs) = x["warm"], x["cold"]
        rw, rc = x["cudnn_warm"] / wm, x["cudnn_cold"] / cm
        cls = []
        if x.get("best_warm"):
            cls.append("best-warm")
        if x.get("best_cold"):
            cls.append("best-cold")
        body.append(
            f"<tr class='{' '.join(cls)}' data-l='{L}'>"
            f"<td>{L}</td><td>{x['BLOCK_N']}</td><td>{x['num_warps']}</td><td>{x['P']}</td>"
            f"<td data-v='{wm}'>{num(wm)}<span class='sd'>±{num(ws)}</span></td>"
            f"<td data-v='{rw}' class='{'win' if rw >= 1 else 'lose'}'>{num(rw)}x</td>"
            f"<td data-v='{cm}'>{num(cm)}<span class='sd'>±{num(cs)}</span></td>"
            f"<td data-v='{rc}' class='{'win' if rc >= 1 else 'lose'}'>{num(rc)}x</td>"
            f"<td>{num(kv_gbps(L, cm), 0)}</td>"
            f"<td data-v='{x['err']}'>{x['err']:.1e}</td>"
            f"<td class='{'ok' if x['ok'] else 'fail'}'>{'geçti' if x['ok'] else 'KALDI'}</td></tr>"
        )

    cudnn_rows = "".join(
        f"<tr><td>{L}</td><td>{num(c['warm'][0])}<span class='sd'>±{num(c['warm'][1])}</span></td>"
        f"<td>{num(c['cold'][0])}<span class='sd'>±{num(c['cold'][1])}</span></td>"
        f"<td>{num(kv_gbps(L, c['cold'][0]), 0)}</td><td>{c['err']:.1e}</td></tr>"
        for L, c in cudnn.items()
    )
    skipped_txt = ", ".join(f"L={L} BN={BN} w={W} P={P}" for L, BN, W, P in skipped)
    l_options = "".join(f"<option value='{L}'>{L}</option>" for L in SEQ_LENS)
    n_ok = sum(r["ok"] for r in rows)

    page = f"""<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GQA Attention Sweep</title>
<style>
:root {{
  --bg: #fbfbfa; --surface: #ffffff; --text: #0b0b0b; --muted: #52514e; --line: #e4e3df;
  --win: #157a52; --lose: #b3401b; --hl-warm: #e8f1fc; --hl-cold: #fdf0e8; --head: #f3f2ef;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #141413; --surface: #1c1c1b; --text: #f0efeb; --muted: #a3a29d; --line: #33322f;
    --win: #4cc792; --lose: #f08a62; --hl-warm: #1d2a3a; --hl-cold: #3a261b; --head: #232321;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #141413; --surface: #1c1c1b; --text: #f0efeb; --muted: #a3a29d; --line: #33322f;
  --win: #4cc792; --lose: #f08a62; --hl-warm: #1d2a3a; --hl-cold: #3a261b; --head: #232321;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--text);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 32px 16px 64px; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }}
h2 {{ font-size: 16px; margin: 36px 0 8px; }}
p, li {{ color: var(--muted); }}
.meta {{ color: var(--muted); font-size: 13px; }}
.wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: var(--surface); }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
th, td {{ padding: 6px 10px; text-align: right; border-bottom: 1px solid var(--line); white-space: nowrap; }}
th {{ background: var(--head); font-weight: 600; position: sticky; top: 0; }}
#full th {{ cursor: pointer; user-select: none; }}
#full th.sorted::after {{ content: attr(data-dir); color: var(--muted); margin-left: 4px; }}
tbody tr:last-child td {{ border-bottom: none; }}
.sd {{ color: var(--muted); font-size: 11px; margin-left: 3px; }}
.cfg {{ color: var(--muted); }}
.win {{ color: var(--win); }}
.lose {{ color: var(--lose); }}
.ok {{ color: var(--win); }}
.fail {{ color: var(--lose); font-weight: 700; }}
tr.best-warm td:first-child {{ box-shadow: inset 4px 0 0 #d63a3a; }}
tr.best-cold td:first-child {{ box-shadow: inset 4px 0 0 #2a78d6; }}
tr.best-warm.best-cold td:first-child {{ box-shadow: inset 4px 0 0 #d63a3a, inset 8px 0 0 #2a78d6; }}
.controls {{ display: flex; gap: 16px; align-items: center; margin: 8px 0; flex-wrap: wrap; color: var(--muted); }}
select {{ font: inherit; padding: 2px 6px; background: var(--surface); color: var(--text);
  border: 1px solid var(--line); border-radius: 6px; }}
.legend span {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin: 0 4px 0 10px; vertical-align: -1px; }}
</style>
</head>
<body>
<main>
<h1>GQA decode attention: parametre taraması</h1>
<div class="meta">{esc(props.name)} · {props.multi_processor_count} SM · L2 {props.L2_cache_size // 2**20} MB ·
torch {esc(torch.__version__)} · triton {esc(triton.__version__)} ·
{datetime.datetime.now():%Y-%m-%d %H:%M} · süre {elapsed:.0f} s</div>

<p>{H_Q} q / {H_KV} kv head, D={D}, BLOCK_G={BLOCK_G}, tek query token, fp16.
Süreler çağrı başına µs (load + merge kernel'i birlikte), CUDA graph içinde ölçüldü.
Bütün graph'lar tek bir random sırada {WARMUP_ITERS} kez ısıtıldı, sonra her biri {ITERS} kez
random sırada ölçüldü; değer ortalama ± std.
<b>warm</b>: hep aynı (q, k, v) seti, K/V L2'de. <b>cold</b>: L2'nin {COLD_FOOTPRINT_L2_MULTIPLE} katı
farklı set random sırada, K/V DRAM'den. Oran = cuDNN / triton (&gt; 1: triton daha hızlı).
GB/s: kv head'lerin K+V byte'ı / cold süre.
Doğruluk: assert_close(triton, cudnn, rtol=1e-2, atol=1e-2) ve fp32 referansa göre max abs hata.
{len(rows)} konfigürasyonun {n_ok}'i geçti.</p>

<h2>L başına en iyi konfigürasyon</h2>
<div class="wrap"><table>
<thead><tr><th>L</th><th>en iyi warm</th><th>BN / warps / P</th><th>cudnn warm</th><th>oran</th>
<th>en iyi cold</th><th>BN / warps / P</th><th>cudnn cold</th><th>oran</th><th>cold GB/s</th></tr></thead>
<tbody>{''.join(summary)}</tbody></table></div>

<h2>cuDNN (SDPA, enable_gqa=True)</h2>
<div class="wrap"><table>
<thead><tr><th>L</th><th>warm</th><th>cold</th><th>cold GB/s</th><th>max abs err</th></tr></thead>
<tbody>{cudnn_rows}</tbody></table></div>

<h2>Bütün konfigürasyonlar</h2>
<div class="controls">
<label>L: <select id="lsel"><option value="">hepsi</option>{l_options}</select></label>
<span>Sıralamak için sütun başlığına tıkla.</span>
<span class="legend"><span style="background:#d63a3a"></span>L için en iyi warm
<span style="background:#2a78d6"></span>L için en iyi cold</span>
</div>
<div class="wrap"><table id="full">
<thead><tr><th>L</th><th>BLOCK_N</th><th>warps</th><th>P</th><th>warm µs</th><th>cudnn/tri warm</th>
<th>cold µs</th><th>cudnn/tri cold</th><th>cold GB/s</th><th>max abs err</th><th>doğruluk</th></tr></thead>
<tbody>{''.join(body)}</tbody></table></div>

<h2>Elenen konfigürasyonlar (P &gt; ceil(L / BLOCK_N))</h2>
<p>{len(skipped)} adet: {esc(skipped_txt) or 'yok'}</p>
</main>
<script>
(function () {{
  const table = document.getElementById("full");
  const tbody = table.tBodies[0];
  const ths = table.tHead.rows[0].cells;
  let col = -1, asc = true;
  function val(td) {{
    const v = td.dataset.v !== undefined ? td.dataset.v : td.textContent;
    const n = parseFloat(v);
    return isNaN(n) ? v : n;
  }}
  Array.from(ths).forEach(function (th, i) {{
    th.addEventListener("click", function () {{
      asc = col === i ? !asc : true;
      col = i;
      Array.from(ths).forEach(function (h) {{ h.classList.remove("sorted"); }});
      th.classList.add("sorted");
      th.dataset.dir = asc ? "▲" : "▼";
      const rows = Array.from(tbody.rows);
      rows.sort(function (a, b) {{
        const x = val(a.cells[i]), y = val(b.cells[i]);
        return (x < y ? -1 : x > y ? 1 : 0) * (asc ? 1 : -1);
      }});
      rows.forEach(function (r) {{ tbody.appendChild(r); }});
    }});
  }});
  document.getElementById("lsel").addEventListener("change", function (e) {{
    const L = e.target.value;
    Array.from(tbody.rows).forEach(function (r) {{
      r.style.display = !L || r.dataset.l === L ? "" : "none";
    }});
  }});
}})();
</script>
</body>
</html>
"""
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(page)


if __name__ == "__main__":
    main()
