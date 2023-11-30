#!/bin/bash

#help
display_help()
{
   echo "DINO by @clementgrisi"
   echo
   echo "Syntax: docker run dino [-p level] [-c filename] [-w wandb_api_key]"
   echo "options:"
   echo "-p     starts self-supervised pre-training at given level (can be either 'patch' or 'region')"
   echo "-c     specify config filename"
   echo "-w     specify wandb API key"
   echo
}

#main
while getopts ":t:p:c:h" opt; do
  case $opt in
    h)
      display_help
      exit 1
      ;;
    p)
      level="$OPTARG"
      ;;
    c)
      config="$OPTARG"
      ;;
    w)
      key="$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

if [[ -n $key ]]; then
  export WANDB_API_KEY=$key
fi

if [[ -n $level ]]; then
  case $level in
    patch)
      python3 -m torch.distributed.run --standalone --nproc_per_node=gpu dino/patch.py --config-name "$config"
      ;;
    region)
      python3 -m torch.distributed.run --standalone --nproc_per_node=gpu dino/region.py --config-name "$config"
      ;;
    *)
      echo "Invalid level option. Please use 'patch' or 'region'"
  esac
else
  echo "No flag specified. Please use -p flag."
fi