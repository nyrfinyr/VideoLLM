# Hydra — gestione configurazioni

Questo progetto usa [Hydra](https://hydra.cc) per la configurazione degli esperimenti. Hydra compone la config finale a partire da più YAML, permette override da CLI e supporta i multirun (grid search senza scrivere loop).

---

## Quando usare Hydra

- **Esperimenti** con varianti di modello / generation / dataset → ottimo (è il caso d'uso target).
- **Script utility one-shot** → over-engineering, basta `argparse` o costanti.
- **Notebook / test** → si usa l'API programmatica `compose()`, non il decorator.

---

## Layout del progetto

```
conf/
├── config.yaml                # entry point: defaults list + valori top-level
├── model/
│   └── qwen_vl.yaml
├── generation/
│   ├── precise.yaml
│   ├── balanced.yaml
│   └── creative.yaml
└── dataset/
    └── egoschema.yaml
```

Ogni sottocartella di `conf/` è un **config group**: contiene varianti mutuamente esclusive di uno stesso "slot" della config.

---

## `config.yaml` — defaults list

```yaml
defaults:
  - model: qwen_vl
  - generation: balanced
  - dataset: egoschema
  - _self_

seed: 42
```

La `defaults list` dice a Hydra **quali file comporre**:

- `model: qwen_vl` → carica `conf/model/qwen_vl.yaml` sotto la chiave `cfg.model`.
- `generation: balanced` → carica `conf/generation/balanced.yaml` sotto `cfg.generation`.
- `dataset: egoschema` → carica `conf/dataset/egoschema.yaml` sotto `cfg.dataset`.
- `_self_` → applica i valori scritti direttamente in `config.yaml` (es. `seed: 42`).

L'ordine conta: chi viene **dopo** sovrascrive chi viene prima. Mettere `_self_` per ultimo significa "i valori in `config.yaml` vincono sui gruppi"; metterlo per primo significa il contrario.

La `cfg` finale è equivalente a:

```yaml
model:
  name: qwen_vl
  torch_dtype: float16
  device_map: auto
generation:
  do_sample: true
  temperature: 0.7
  top_p: 0.9
  max_new_tokens: 512
  repetition_penalty: 1.1
dataset:
  name: egoschema
  root: /home/alesvale/datasets/egoschema_subset_20
  fps: 0.25
seed: 42
```

---

## Entry point: `main.py`

```python
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)
```

Il decorator `@hydra.main`:

1. legge `sys.argv` per gli override CLI;
2. carica `conf/config.yaml` e risolve la `defaults list`;
3. applica gli override CLI;
4. crea la working dir (`outputs/YYYY-MM-DD/HH-MM-SS/`) e salva la config risolta in `.hydra/config.yaml`;
5. chiama la funzione decorata passandole il `DictConfig` come primo argomento.

Tu chiami `main()` senza argomenti — è il decorator che passa `cfg`.

### Separazione `main` / `run`

La logica vera vive in `run(cfg)`, una funzione pura. `main` è solo il wrapper Hydra. Vantaggi:

- `run` è chiamabile da notebook / test / script con un `cfg` costruito a mano (vedi sotto).
- `main` resta minimale e non mischia concerns.

---

## Comandi tipici

### Run di default
```bash
python main.py
```

### Cambiare un config group (variante)
```bash
python main.py generation=creative
python main.py generation=precise model=qwen_vl
```

### Override puntuale di un singolo campo
```bash
python main.py generation.temperature=0.3
python main.py run.prompt="Cosa vedi?" seed=7
```

Combinabili:
```bash
python main.py generation=creative generation.temperature=0.5 seed=0
```

### Multirun (sweep)
```bash
# 3 run con preset diversi
python main.py -m generation=precise,balanced,creative

# grid: 3 temperature × 2 seed = 6 run
python main.py -m generation.temperature=0.3,0.7,1.1 seed=0,1
```

I risultati finiscono in `multirun/YYYY-MM-DD/HH-MM-SS/<index>/`.

### Stampare la config risolta senza eseguire
```bash
python main.py --cfg job        # solo la job config (cfg passata a main)
python main.py --cfg hydra      # config interna di Hydra
python main.py --help           # help generato + valori correnti
```

### Cambiare working dir
```bash
python main.py hydra.run.dir=outputs/$(date +%F)/exp_qwen_creative
```

---

## Working dir e riproducibilità

Ogni invocazione Hydra:

1. crea una nuova directory (default `outputs/YYYY-MM-DD/HH-MM-SS/`);
2. **fa `chdir` in quella directory** prima di chiamare `main`;
3. salva al suo interno `.hydra/config.yaml` (config risolta), `.hydra/overrides.yaml` (override CLI usati), `.hydra/hydra.yaml` (config interna).

Conseguenze pratiche:

- I **path relativi** dentro `run(cfg)` non puntano più alla root del repo. Usa path assoluti, o `hydra.utils.get_original_cwd()` per recuperare la cwd da cui hai lanciato.
- I file di output (log, checkpoint, plot) vanno bene scritti relativi: finiscono ordinati per run nella working dir.
- Per riprodurre un esperimento basta rileggere il `.hydra/config.yaml` di quella cartella.

---

## Da dict (YAML) a oggetti Python

`cfg` è un `DictConfig` di OmegaConf: si accede come attributo (`cfg.model.device_map`) o come dict (`cfg["model"]["device_map"]`). Non è però un `dict` Python puro — alcune librerie esterne si confondono. In quei casi:

```python
plain = OmegaConf.to_container(cfg.generation, resolve=True)   # dict puro
gen_cfg = GenerationConfig(**plain)
```

`resolve=True` espande le eventuali interpolazioni (`${...}`).

### Stringhe → tipi non-YAML

YAML non sa esprimere `torch.float16`. Pattern adottato: la stringa nel YAML, la mappa nel codice.

```python
_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
torch_dtype = _DTYPES[cfg.model.torch_dtype]
```

Stesso pattern per scegliere la classe modello a partire da `cfg.model.name`.

---

## Uso da notebook / test (API programmatica)

Il decorator `@hydra.main` non si usa fuori dal main del processo (cambierebbe cwd e leggerebbe `sys.argv`). Per usare la stessa config in un notebook o in un test:

```python
from hydra import initialize, compose
from main import run

with initialize(version_base=None, config_path="conf"):
    cfg = compose(
        config_name="config",
        overrides=["generation=creative", "generation.temperature=0.5"],
    )

run(cfg)        # logica pura, niente magic
```

`initialize` usa un context manager che setta il search path senza alterare cwd. `compose` ritorna il `DictConfig` come lo avresti da CLI.

---

## Aggiungere una nuova variante

Esempio: nuovo preset di generazione `greedy_long`.

1. Crea `conf/generation/greedy_long.yaml`:
   ```yaml
   do_sample: false
   num_beams: 1
   max_new_tokens: 2048
   repetition_penalty: 1.05
   ```
2. Usalo: `python main.py generation=greedy_long`.

Esempio: nuovo modello (es. LLaVA).

1. Implementa la sottoclasse di `BaseVLM` in `models/llava.py`.
2. Registrala nel dict `_MODELS` in `main.py`:
   ```python
   _MODELS = {"qwen_vl": QwenVL, "llava": LLaVA}
   ```
3. Crea `conf/model/llava.yaml` con `name: llava` + i parametri.
4. Usa: `python main.py model=llava`.

---

## Gotcha frequenti

- **`_self_` mancante o nel posto sbagliato**: i valori top-level di `config.yaml` non si vedono o vengono sovrascritti dai gruppi. Tienilo in fondo alla `defaults list` salvo motivi specifici.
- **Override di un campo che non esiste**: errore `Could not override 'foo.bar'`. Hydra è strict di default — i campi devono esistere già in qualche YAML. Per aggiungerne al volo: `+foo.bar=42`. Per forzare la sostituzione anche se esiste: `++foo.bar=42`.
- **Stringhe con spazi / caratteri speciali**: in shell devi quotare. `python main.py run.prompt="Cosa vedi nell'immagine?"`.
- **Working dir cambiata**: se il tuo codice apre un file con path relativo (es. `open("data.csv")`), non lo trova. Soluzione: `hydra.utils.get_original_cwd()` oppure path assoluti.
- **`DictConfig` ≠ `dict`**: vedi sopra, usa `OmegaConf.to_container` per librerie che vogliono dict puro.
- **Notebook**: usa `initialize`/`compose`, non il decorator.

---

## Riferimenti

- Hydra docs: <https://hydra.cc/docs/intro/>
- OmegaConf docs: <https://omegaconf.readthedocs.io/>
- Structured Configs (validazione tipi via dataclass): <https://hydra.cc/docs/tutorials/structured_config/intro/>
