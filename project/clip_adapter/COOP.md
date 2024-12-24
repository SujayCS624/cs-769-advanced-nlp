## How to Run

Run `clip_adapter/scripts/coop/main.sh`, which contains six input arguments.

`DATASET` takes as input a dataset name, like `imagenet` or `caltech101`. The valid names are the files' names in `clip_adapter/configs/datasets/`.

`CFG` means which config file to use. For now, we have implemented just `rn50` (options in `clip_adapter/configs/trainers/clip_adapter/`).

`CTP` determines where the class token position is present in the prompt. It can be either `end` or `middle`. We have placed this at the end.

`NCTX` determines the number of context tokens. We have used the value of `16`.

`SHOTS` determines the number of labelled training examples used per class. We have varied this parameter in the following range `(1, 2, 4, 8, 16)`

`CSC` determines if we are providing any class-specific context. For the case of the `rsna_pneumonia`, this is set to `False`.

Below are the commands we used to run CoOp on the RSNA Pneumonia dataset.

**COOP**:
- 1 shot: `bash scripts/clip_adapter/main.sh rsna_pneumonia rn50 end 16 1 False`
- 2 shots: `bash scripts/clip_adapter/main.sh rsna_pneumonia rn50 end 16 2 False`
- 4 shots: `bash scripts/clip_adapter/main.sh rsna_pneumonia rn50 end 16 4 False`
- 8 shots: `bash scripts/clip_adapter/main.sh rsna_pneumonia rn50 end 16 8 False`
- 16 shots: `bash scripts/clip_adapter/main.sh rsna_pneumonia rn50 end 16 16 False`

After the experiments are finished, you can run `parse_results.py` to calculate the average results instead of manually looking into the log files. 

To calculate the average results for the folder `rn50_16shots/nctx16_cscFalse_ctpend/`, you can run

```bash
python parse_results.py output/rsna_pneumonia/CoOp/rn50_16shots/nctx16_cscFalse_ctpend
```