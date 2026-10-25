#!/usr/bin/env bash
# Usage:  bash run_experiments.sh stage1|stage2|stage3|bonus|all
# Optional env vars: EPOCHS (default 20)  MAX_TRAIN (subsample, e.g. 20000)  BS_BASE LR_BASE
# stage3 needs the best hyperparameters you found:  BEST_LR=1e-3 BEST_OPT=adam BEST_INIT=xavier BEST_BS=64 bash run_experiments.sh stage3
set -e
EPOCHS=${EPOCHS:-20}
COMMON="--epochs $EPOCHS ${MAX_TRAIN:+--max_train $MAX_TRAIN}"
HID=512; PROJ=256; LAYERS=2

stage1() {   # same settings for every architecture
  python train.py --cell lstm  --hid $HID --layers $LAYERS --name main_lstm  $COMMON
  python train.py --cell cifg  --hid $HID --layers $LAYERS --name main_cifg  $COMMON
  python train.py --cell lstmp --hid $HID --proj $PROJ --layers $LAYERS --name main_lstmp $COMMON
}
stage2() {   # one factor at a time on the vanilla baseline
  for lr in 1e-2 1e-3 1e-4; do python train.py --cell lstm --lr $lr --name hp_lr_$lr $COMMON; done
  for o in adam adamw rmsprop sgd; do python train.py --cell lstm --opt $o --name hp_opt_$o $COMMON; done
  for i in default xavier orthogonal; do python train.py --cell lstm --init $i --name hp_init_$i $COMMON; done
  for b in 32 64 128; do python train.py --cell lstm --bs $b --name hp_bs_$b $COMMON; done
}
stage3() {   # re-run the 3 architectures with the best hyperparameters
  F="--lr ${BEST_LR:-1e-3} --opt ${BEST_OPT:-adam} --init ${BEST_INIT:-xavier} --bs ${BEST_BS:-64}"
  python train.py --cell lstm  --name final_lstm  $F $COMMON
  python train.py --cell cifg  --name final_cifg  $F $COMMON
  python train.py --cell lstmp --proj $PROJ --name final_lstmp $F $COMMON
}
bonus() {
  python train.py --cell wmc --name bonus_wmc $COMMON
  python train.py --cell lstm --attn --name bonus_lstm_attn $COMMON
}
case "$1" in
  stage1) stage1;; stage2) stage2;; stage3) stage3;; bonus) bonus;;
  all) stage1; stage2; stage3; bonus;;
  *) echo "usage: $0 stage1|stage2|stage3|bonus|all"; exit 1;;
esac
python aggregate.py
