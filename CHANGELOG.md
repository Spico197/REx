# Changelog

## Working on

- v0.1.9beta
  - change logo image
  - add `CachedTransformBase` and `CachedTransformOneBase`
  - remove `register` decorator in `SimpleTask` and `SimpleMetricTask`, so that code completion could be hinted
  - remove param type in `register_on_namespace`
  - change the return value of `find_all_positions` from `List[List[int]]` to `List[Tuple[int]]`
  - add `dataset_name` argument when calling transforming ([#12](https://github.com/Spico197/REx/issues/12))
  - add `is_eval=True` argument to call `model()` when evaluating to indicating it returns predictions ([#16](https://github.com/Spico197/REx/issues/16))
  - change `os.path` into `pathlib.Path` to remove `os` mixup usage
  - move `init_optimizer` and `init_lr_scheduler` into `train` call to alleviate unexpected train set transformation during evaluating ([#15](https://github.com/Spico197/REx/issues/15))
  - add `log_loss` and `log_metrics` interfaces to support aimhubio/tensorboard logging in inherited sub-tasks
  - remove `base_configpath`, use multiple `-c` instead. e.g. `rex train -d -c 1.yaml -c 2.yaml -a a=1 b=2`
  - fix overwrite warning when creating new task directory
  - log debug message to record types of model, data_manager, transform
  - add `gen_conf` command to generate configuration template
  - rename `utils.iteration.flatten_all` to `flatten_all_iter`
  - add `utils.iteration.windowed_queue_iter` to get windowed batch elements
  - add `utils.position.find_element_in_list` to get all indices of an element in a list
  - add `utils.segmentation.split_list_by_element` to split a list into segments by one element
  - integrate Learn-REx example

## Previous versions

- v0.1.8
  - fix wrong call in `MetricBase`
  - change default `ncols` of `pbar` into 100
  - add `dump_middle=True` argument to `.eval()` in `SimpleMetricTask`
  - change logic in `SimpleTask._eval_during_train()` to make sure ckpt dumping
  - dump model at the beginning of `_eval_during_train` if `save_every_ckpt` in case of any unexpected crashes
  - fix mutable arg in `LabelEncoder`: change `initial_dict` into `initial`, default is None
  - add `missing_key_as_null` arg in `GeneralCollateFn`
  - remove mutable default dict value in `key2type` arg in `GeneralCollateFn`
  - move `GeneralCollateFn._valid_data` to `rex.utils.batch/validate_instance_has_the_same_keys()`
  - add `group_instances_into_batch` and `decompose_batch_into_instances` in `rex.utils.batch`
  - fix comment typos in `TaskBase.load()` and `MetricBase.calculate_scores()`
  - add `regenerate_cache` config flag for `DataManager`
  - update template

- v0.1.7
  - add `SimpleMetricTask` for training and evaluation with metric instances
  - add `MetricBase` as a reference base class for `SimpleMetricTask`
  - fix online testing bug when data is not provided
  - show dataset name in cached transforming
  - show the number of model parameters when training
  - add `type_idx=None` to support micro-averaged-only `tagging_f1` scores calculation

- v0.1.6
    - add `warmup_proportion` to default config. ref to [#7](https://github.com/Spico197/REx/issues/7)
    - add `rex version` command
    - fix train loss == eval loss problem in issue [#6](https://github.com/Spico197/REx/issues/6)
    - fix type err in [#9](https://github.com/Spico197/REx/issues/9)
    - add `encode_one` and `decode_one` into `LabelEncoder`
    - add json-friendly dumping type convertion function (as a `dump_json*` built-in convertion)
    - add tests for relation extraction tasks
- v0.1.5: add `update_before_tensorify` into `GeneralCollateFn`, fix logging level displaying problem
- v0.1.4: move accelerate to `rex.__init__`, update multi process tqdm & logging (only show in the main process in default), remove cache in debug mode, fix bugs in `rex.cmds.new`, add `rank_zero_only` in task dump, `load_best_ckpt` if `resumed_training`
- v0.1.3: fix emb import
- v0.1.1: update registry and add `accelerate` multi-gpu support
- v0.1.0: huge update with lots of new features, check the usage in `examples/IPRE` ~
- v0.0.15: add safe_try to kill ugly statements in example main call
- v0.0.14: update vocab embedding loading to be compatible with other embedding files
- v0.0.13: update vocab, label_encoder, fix bugs in cnn reshaping and crf importing
- v0.0.12: fix crf module import issue
- v0.0.11: fix templates data resources
- v0.0.10: update `utils.config` module, `StaticEmbedding` decoupling, remove eps in metrics, add templates generation entrypoints, add more tests (coverage stat for the whole repo, lots of codes are not test case covered)
- v0.0.9: use `argparse` instead of `click`, move loguru logger into `rex.utils.logging`, add hierarchical config setting
- v0.0.8: fix config loading, change default of `makedirs` and `dump_configfile` to be `True`
- v0.0.7: fix recursive import bug
- v0.0.6: integrate omega conf loading into the inner task, add `load_*_data` option to data managers
- v0.0.5: update ffn
- v0.0.4: return detailed classification information in `mc_prf1`, support nested dict tensor movement
- v0.0.3: fix packaging bug in `setup.py`
- v0.0.2: add black formatter and pytest testing
- v0.0.1: change `LabelEncoder.to_binary_labels` into `convert_to_multi_hot` or `convert_to_one_hot`
