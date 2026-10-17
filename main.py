import atexit
import itertools
import os
import random
import statistics
import time

import torch
import torch.cuda.nvtx as nvtx
from transformers import AutoTokenizer, AutoModelForCausalLM

from attn import replace_attention
from kv_cache.manager import KVCacheManager

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# KV cache kapasitesi (token): manager bir kez bu boyutta ayrılıyor, decode graph'ı tek
# (her prompt uzunluğu aynı graph'ı kullanır, kernel geçerli uzunluğu GPU'dan okuyor).
# Qwen2.5'in max_position_embeddings'i. fp16'da 24 katman x K+V: ~12 KB/token, ~400 MB
KV_CACHE_CAPACITY = 32768


def get_eos_token_ids(model, tokenizer):
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    if isinstance(eos, int):
        eos = [eos]
    return set(eos)


def tokenize(tokenizer, messages, device):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(device)


# manuel decode: torch.compile(inductor) ile derlenmiş forward, attention attn.py'deki
# Triton GQA kernel'i (custom op olarak), hepsi tek CUDA graph.
# prefill eager, attention sdpa. KV cache: kv_cache.KVCacheManager.
# vllm karşılaştırma ve doğruluk referansı (sadece Linux/WSL'de kurulu)
IMPL_LABELS = {
    "vllm": "vLLM (LLMEngine, varsayılan ayarlar)",
    "manual_inductor_triton_attn": "manual + inductor + Triton attention + CUDA graph",
}

# vLLM'in model için ayırabileceği GPU belleği oranı (ağırlıklar + KV cache + graph'lar).
# aynı GPU'da bizim model ve KV cache'lerimiz de duruyor, varsayılan 0.92 sığmıyor
VLLM_GPU_MEMORY_UTILIZATION = 0.4


def use_triton_attention(model):
    # modelin Qwen2Attention'larını kalıcı olarak TritonQwen2Attention ile değiştirir
    # (ilk çağrıda, sonra no-op). forward'a kv_manager verilince K/V'yi manager'a yazar,
    # decode'da Triton kernel'i, prefill'de sdpa kullanır; kv_manager'sız orijinal modüle
    # gider. eskiden context manager'dı (hf generate aynı modeli orijinal attention'la
    # kullanıyordu), artık modeli başka kullanan yok
    if not getattr(model, "_uses_triton_attention", False):
        replace_attention(model)
        model._uses_triton_attention = True


class DecodeGraph:
    # batch_size dizinin birer token'lık decode forward'unu bir kez CUDA graph olarak
    # kaydeder, sonraki generate çağrılarında aynı buffer'lar ve cache ile tekrar kullanılır.
    # cache dizi başına max_len kapasiteli, max_len'e kadar her prompt uzunluğu bu graph'la
    # çalışır. batch boyutu graph'ın şekillerine gömülü: her batch boyutu ayrı graph

    def __init__(self, model, max_len, batch_size=1, warmup_iters=3):
        self.model = model
        self.batch_size = batch_size
        self.max_len = max_len
        device = model.device

        with torch.inference_mode():
            self.static_token = torch.zeros(
                (batch_size, 1), device=device, dtype=torch.long
            )
            self.static_position_ids = torch.zeros(
                (batch_size, 1), device=device, dtype=torch.long
            )
            # buffer'lar burada bir kez ayrılıyor, adresleri sabit (graph bunları kaydediyor)
            self.kv_cache = KVCacheManager(
                model.config, max_len, batch_size, device, model.dtype
            )

            # attention'lar Triton'a çevrilmiş olmalı (use_triton_attention):
            # graph capture sırasında çalışan kernel'leri kaydeder
            use_triton_attention(model)

            # capture'dan önce yan stream'de warmup: cuBLAS workspace, triton JIT,
            # inductor derlemesi gibi ilk çalıştırmada olan işler capture'a girmesin
            side_stream = torch.cuda.Stream()
            side_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side_stream):
                self.kv_cache.reset()
                self._forward()

                # mode varsayılan (CUDA graph'sız): graph'ı aşağıda biz
                # alıyoruz, inductor'ın triton kernel'leri ona kaydedilir
                forward = torch.compile(
                    self._forward, fullgraph=True, dynamic=False
                )

                for _ in range(warmup_iters):
                    self.kv_cache.reset()
                    forward()
            torch.cuda.current_stream().wait_stream(side_stream)

            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_logits = forward().logits

    def _forward(self):
        # use_cache=False: hf kendi cache'ini oluşturmuyor, attention'lar K/V'yi
        # kv_manager'a yazıyor. mask verilmiyor (Triton attention causal + pad'siz)
        outputs = self.model(
            input_ids=self.static_token,
            position_ids=self.static_position_ids,
            kv_manager=self.kv_cache,
            logits_to_keep=1,
            use_cache=False
        )
        # bütün katmanlar yazdıktan sonra bir kez: graph'a kaydedilir, her replay
        # bütün dizilerde +1
        self.kv_cache.advance(1)
        return outputs

    def replay(self):
        with nvtx.range("manual_graph_replay"):
            self.graph.replay()
        return self.static_logits


_decode_graphs = {}


# batch'li decode graph'larının batch boyutları (kovalar). gerçek batch boyutu kendisinden
# büyük ilk kovaya yuvarlanır, fazla slotlar sahte dizilerle doldurulur. küçük batch'te
# decode ağırlık okumaya bağlı, fazladan birkaç dizi adım süresini pek değiştirmiyor
DECODE_BATCH_BUCKETS = (1, 2, 4, 8, 16)


def decode_bucket(batch_size):
    for bucket in DECODE_BATCH_BUCKETS:
        if batch_size <= bucket:
            return bucket
    raise ValueError(
        f"batch size {batch_size} en büyük kovadan ({DECODE_BATCH_BUCKETS[-1]}) büyük"
    )


def get_decode_graph(model, batch_size=1):
    # (model, batch boyutu) başına bir graph (ilk çağrıda derleme + capture).
    # toplam cache her graph'ta KV_CACHE_CAPACITY token (~400 MB): dizi başına kapasite
    # KV_CACHE_CAPACITY // batch_size
    key = (model, batch_size)
    if key not in _decode_graphs:
        _decode_graphs[key] = DecodeGraph(
            model, KV_CACHE_CAPACITY // batch_size, batch_size
        )
    return _decode_graphs[key]


def generate(model, inputs, eos_token_ids, max_new_tokens=10, on_token=None):
    if inputs["input_ids"].shape[0] != 1:
        raise ValueError("generate sadece batch size 1 destekliyor")

    if max_new_tokens <= 0:
        return []

    # Triton attention mask kullanmıyor, ilk seq_len pozisyonun hepsine bakıyor
    if not inputs["attention_mask"].bool().all():
        raise ValueError("generate prompt'ta pad desteklemiyor")

    device = inputs["input_ids"].device
    prompt_len = inputs["input_ids"].shape[-1]

    if prompt_len + max_new_tokens > KV_CACHE_CAPACITY:
        raise ValueError(
            f"prompt + yeni token ({prompt_len} + {max_new_tokens}) KV cache "
            f"kapasitesini ({KV_CACHE_CAPACITY}) aşıyor"
        )

    generated_ids = torch.empty(
        (1, max_new_tokens), device=device, dtype=torch.long
    )
    num_generated = 0

    with torch.inference_mode():
        # prefill de K/V'yi manager'a TritonQwen2Attention üzerinden yazıyor
        use_triton_attention(model)
        decode_graph = get_decode_graph(model)
        kv_cache = decode_graph.kv_cache

        # yeni istek: cache baştan (prefill pozisyonları 0..prompt_len-1)
        kv_cache.reset()

        outputs = model(
            input_ids=inputs["input_ids"],
            position_ids=torch.arange(prompt_len, device=device)[None],
            kv_manager=kv_cache,
            logits_to_keep=1,
            use_cache=False
        )
        kv_cache.advance(prompt_len)
        logits = outputs.logits

        with nvtx.range("manual_decode"):
            for step in range(max_new_tokens):
                with nvtx.range("argmax"):
                    next_token = torch.argmax(
                        logits[:, -1, :],
                        dim=-1,
                        keepdim=True
                    )

                with nvtx.range("eos_check"):
                    is_eos = next_token.item() in eos_token_ids

                if on_token is not None:
                    on_token()

                if is_eos:
                    break

                with nvtx.range("store_token"):
                    generated_ids[:, step] = next_token[:, 0]

                # decode pozisyonları prefill'in (0..prompt_len-1) devamı; cache'e yazma
                # pozisyonu manager.start, ikisi aynı ilerliyor
                position = prompt_len + num_generated
                num_generated += 1

                if step == max_new_tokens - 1:
                    break

                decode_graph.static_token.copy_(next_token)
                decode_graph.static_position_ids.fill_(position)

                with nvtx.range("decode_forward"):
                    logits = decode_graph.replay()

    return generated_ids[0, :num_generated].tolist()


def generate_batch(model, prompts, eos_token_ids, max_new_tokens=10, on_step=None):
    # static batching: prompts (token id listeleri) birlikte prefill edilir, sonra her
    # adımda tek graph replay'i bütün dizilere birer token üretir. biten dizi graph'tan
    # çıkamıyor (şekil sabit), çalışmaya devam ediyor ama token'ları atılıyor. döngü
    # bütün diziler bitince ya da max_new_tokens'te durur.
    # on_step: adım başına bir kez (ilk çağrı prefill'den sonra, ilk token'lar seçilince):
    # aynı adımın token'ları birlikte çıkıyor, TPOT her dizi için aynı.
    # dönüş: dizi başına üretilen token'lar (eos hariç)
    B = len(prompts)
    if B == 0 or max_new_tokens <= 0:
        return [[] for _ in prompts]
    if any(len(p) == 0 for p in prompts):
        raise ValueError("boş prompt desteklenmiyor")

    bucket = decode_bucket(B)
    lens_list = [len(p) for p in prompts]
    T = max(lens_list)

    with torch.inference_mode():
        use_triton_attention(model)
        decode_graph = get_decode_graph(model, bucket)
        kv_cache = decode_graph.kv_cache
        device = model.device

        # bütün diziler aynı sayıda adım ilerliyor (biten de): en uzun prompt sığmalı
        if T + max_new_tokens > decode_graph.max_len:
            raise ValueError(
                f"prompt + yeni token ({T} + {max_new_tokens}) dizi başına KV cache "
                f"kapasitesini ({decode_graph.max_len}, batch {bucket}) aşıyor"
            )

        # right padding: kernel sadece uzunluk biliyor, geçerli token'lar her dizinin başında
        # olmalı. kovanın fazla slotları tek token'lık sahte dizi (L >= 1, kernel'de
        # boş dizi yok), çıktıları kullanılmıyor
        input_ids = torch.zeros((bucket, T), dtype=torch.long)
        lens = torch.ones(bucket, dtype=torch.long)
        for b, p in enumerate(prompts):
            input_ids[b, :len(p)] = torch.tensor(p)
            lens[b] = len(p)
        input_ids, lens = input_ids.to(device), lens.to(device)

        kv_cache.reset()

        # lm_head sadece her dizinin son gerçek token'ında: bütün pozisyonların
        # logit'i (bucket x T x 151936) uzun prompt'ta GB'larca olurdu.
        # pad'ler causal mask yüzünden gerçek token'ları etkilemiyor
        hidden = model.model(
            input_ids=input_ids,
            position_ids=torch.arange(T, device=device)[None].expand(bucket, T),
            kv_manager=kv_cache,
            use_cache=False
        ).last_hidden_state
        kv_cache.advance(lens)
        logits = model.lm_head(hidden[torch.arange(bucket, device=device), lens - 1])[:, None]

        outputs = [[] for _ in range(B)]
        finished = [False] * B

        with nvtx.range("manual_batch_decode"):
            for step in range(max_new_tokens):
                next_tokens = torch.argmax(logits[:, -1, :], dim=-1)

                # host'a tek kopya (adım başına bir sync, B=1 generate'teki .item() gibi)
                tokens = next_tokens[:B].tolist()

                if on_step is not None:
                    on_step()

                for b, token in enumerate(tokens):
                    if finished[b]:
                        continue
                    if token in eos_token_ids:
                        finished[b] = True
                    else:
                        outputs[b].append(token)

                if all(finished) or step == max_new_tokens - 1:
                    break

                # her dizinin yeni token'ı kendi pozisyonunda: manager.start (cache'e
                # yazma pozisyonu) ile aynı, graph replay sonunda +1
                decode_graph.static_token.copy_(next_tokens[:, None])
                decode_graph.static_position_ids.copy_(kv_cache.start[:, None])

                with nvtx.range("decode_forward"):
                    logits = decode_graph.replay()

    return outputs


_vllm_engine = None
_vllm_request_ids = itertools.count()


def get_vllm_engine():
    # ilk çağrıda oluşturulur (~1-2 dk: torch.compile + CUDA graph capture), sonra
    # aynı engine kullanılır. vllm import'u da burada: Windows'ta kurulu değil
    global _vllm_engine
    if _vllm_engine is None:
        # FlashInfer'ın top-k/top-p sampler'ı ilk çalışmada nvcc ile JIT derleniyor,
        # WSL'de CUDA toolkit yok. greedy decode onu zaten kullanmıyor
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLMEngine, EngineArgs

        # dtype: Qwen'in config'i bf16, bizim model fp16 yükleniyor.
        # prefix caching kapalı: benchmark aynı prompt'u tekrar tekrar veriyor, açık
        # olsa prefill cache'ten gelir ve TTFT haksız biçimde düşer.
        # gerisi vLLM varsayılanı (engine core ayrı süreçte, FlashAttention 2,
        # torch.compile + CUDA graph)
        _vllm_engine = LLMEngine.from_engine_args(EngineArgs(
            model=MODEL_ID,
            dtype="float16",
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
            enable_prefix_caching=False,
        ))
        # kapatılmazsa engine core süreci yaşamaya devam ediyor, program çıkışta takılıyor
        atexit.register(_vllm_engine.engine_core.shutdown)
    return _vllm_engine


def generate_vllm(model, inputs, eos_token_ids, max_new_tokens=10, on_token=None):
    # model kullanılmıyor (vLLM ağırlıkları kendisi yüklüyor), imza diğerleriyle aynı
    from vllm import SamplingParams, TokensPrompt
    from vllm.sampling_params import RequestOutputKind

    if max_new_tokens <= 0:
        return []

    engine = get_vllm_engine()

    # detokenize kapalı: diğer impl'ler de üretim sırasında metne çevirmiyor.
    # DELTA: her step() sadece yeni token'ları döndürür (decode'da adım başına 1)
    params = SamplingParams(
        temperature=0,
        max_tokens=max_new_tokens,
        stop_token_ids=list(eos_token_ids),
        detokenize=False,
        output_kind=RequestOutputKind.DELTA,
    )
    engine.add_request(
        f"req{next(_vllm_request_ids)}",
        TokensPrompt(prompt_token_ids=inputs["input_ids"][0].tolist()),
        params,
    )

    new_ids = []
    while engine.has_unfinished_requests():
        for output in engine.step():
            token_ids = output.outputs[0].token_ids
            new_ids.extend(token_ids)
            if on_token is not None:
                for _ in token_ids:
                    on_token()

    # eos çıktıda geliyor, diğer impl'ler gibi dönen listeye koymuyoruz
    if new_ids and new_ids[-1] in eos_token_ids:
        new_ids = new_ids[:-1]

    return new_ids


# benchmark'larda karşılaştırılan implementasyonlar, anahtarlar IMPL_LABELS ile aynı
IMPL_FNS = {
    "vllm": generate_vllm,
    "manual_inductor_triton_attn": generate,
}


def measure(generate_fn, model, inputs, eos_token_ids, max_new_tokens):
    token_times = []

    torch.cuda.synchronize()
    start = time.perf_counter()

    ids = generate_fn(
        model, inputs, eos_token_ids, max_new_tokens,
        on_token=lambda: token_times.append(time.perf_counter())
    )

    # eos çıktıda yok ama zamanı ölçüldü, token_times ids'ten bir uzun olur
    hit_eos = len(token_times) > len(ids)

    ttft = token_times[0] - start if len(token_times) >= 1 else None
    tpot = (
        (token_times[-1] - token_times[0]) / (len(token_times) - 1)
        if len(token_times) >= 2 else None
    )

    return ids, hit_eos, ttft, tpot


def run_benchmarks(model, cases, eos_token_ids, max_new_tokens,
                   warmup_iters=3, iters=10, seed=0):
    # cases: {key: (generate_fn, inputs)}
    # tüm (case, iter) çiftleri karıştırılıp öyle koşulur, böylece ısınma,
    # clock değişimi gibi zamana bağlı etkiler tek bir case'e yığılmaz
    rng = random.Random(seed)

    warmup_schedule = [key for key in cases for _ in range(warmup_iters)]
    rng.shuffle(warmup_schedule)
    for key in warmup_schedule:
        generate_fn, inputs = cases[key]
        generate_fn(model, inputs, eos_token_ids, max_new_tokens)

    schedule = [key for key in cases for _ in range(iters)]
    rng.shuffle(schedule)

    results = {
        key: {"ids": None, "consistent": True, "eos": 0, "runs": 0,
              "ttft": [], "tpot": []}
        for key in cases
    }

    for key in schedule:
        generate_fn, inputs = cases[key]
        ids, hit_eos, ttft, tpot = measure(
            generate_fn, model, inputs, eos_token_ids, max_new_tokens
        )

        # aynı girdi her ölçümde aynı token'ları üretmeli
        if results[key]["ids"] is not None and results[key]["ids"] != ids:
            results[key]["consistent"] = False
        results[key]["ids"] = ids
        results[key]["eos"] += hit_eos
        results[key]["runs"] += 1
        if ttft is not None:
            results[key]["ttft"].append(ttft)
        if tpot is not None:
            results[key]["tpot"].append(tpot)

    return results


def summarize(samples):
    if not samples:
        nan = float("nan")
        return {"mean": nan, "median": nan, "std": nan, "min": nan,
                "p25": nan, "p75": nan, "max": nan}

    if len(samples) >= 2:
        p25, _, p75 = statistics.quantiles(samples, n=4, method="inclusive")
        std = statistics.stdev(samples)
    else:
        p25 = p75 = samples[0]
        std = 0.0

    return {
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "std": std,
        "min": min(samples),
        "p25": p25,
        "p75": p75,
        "max": max(samples),
    }


STAT_KEYS = ["mean", "median", "std", "min", "p25", "p75", "max"]


def eos_cell(result):
    # kaç ölçüm eos ile bitti, örn. "10/10"
    return f"{result['eos']}/{result['runs']}"


def stats_header(metric):
    return "".join(f"{metric + '_' + k:>13}" for k in STAT_KEYS)


def stats_row(stats):
    # saniye -> ms
    return "".join(f"{stats[k] * 1000:>13.2f}" for k in STAT_KEYS)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16
    ).to("cuda").eval()

    messages = [
        {"role": "user", "content": "What is a GPU warp?"}
    ]

    max_new_tokens = 50
    warmup_iters = 3
    iters = 10
    seed = 0

    inputs = tokenize(tokenizer, messages, model.device)
    eos_token_ids = get_eos_token_ids(model, tokenizer)

    results = run_benchmarks(
        model,
        {name: (fn, inputs) for name, fn in IMPL_FNS.items()},
        eos_token_ids, max_new_tokens, warmup_iters, iters, seed
    )

    # referans: vLLM
    ref_ids = results["vllm"]["ids"]
    print(f"text: {tokenizer.decode(ref_ids, skip_special_tokens=True)!r}")
    print()
    print(f"{'impl':<28}{'same':>6}{'consistent':>12}  label")
    for name, label in IMPL_LABELS.items():
        r = results[name]
        print(f"{name:<28}{str(r['ids'] == ref_ids):>6}{str(r['consistent']):>12}  {label}")
    print()
    print(f"seq_len={inputs['input_ids'].shape[-1]}  max_new_tokens={max_new_tokens}  "
          f"warmup_iters={warmup_iters}  iters={iters}  seed={seed}  (ms)")
    for metric in ["ttft", "tpot"]:
        print()
        print(f"{'impl':<28}{'new_tok':>9}{'eos':>7}{stats_header(metric)}")
        for name in IMPL_LABELS:
            r = results[name]
            print(
                f"{name:<28}{len(r['ids']):>9}{eos_cell(r):>7}"
                f"{stats_row(summarize(r[metric]))}"
            )
