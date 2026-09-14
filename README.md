# knowledgeFlow

Experiments around shortening Spark-X2.5-1.7B reasoning without dropping accuracy.

Kaggle kernels stay thin: they clone this repo and run Python scripts. Real code lives in `scripts/`.

## Kaggle loop

```bash
git add -A && git commit && git push
kaggle kernels push -p kaggle/spark-reason-probe --accelerator NvidiaTeslaT4
```

The notebook clones `https://github.com/Priyanshu-5257/knowledgeFlow` and runs:

```bash
python scripts/probe_reasoning.py --out /kaggle/working/spark_reason_probe.json
```

Spark-X2.5 needs `transformers==4.57.1`. Newer Transformers crash on Spark's per-layer-type `rope_parameters`. Load the 1.7B model on **one** T4 (FP16 ~3.4 GB). `device_map=auto` across 2xT4 splits weights and breaks `generate()`.
