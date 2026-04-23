#!/usr/bin/env bash

# Source this file to define a5-srun:
#   source a5_srun.sh
#   a5-srun python histogram.py benchmark
#   a5-srun --mem 8G --time 00:20:00 python histogram.py profile

module load cudatoolkit/13.0
module load nsight-systems/2024.4.1

a5-srun() {
    local default_cpus_per_task="2"
    local default_mem="4G"
    local default_gres="gpu:3g.20gb:1"
    local default_gpu_freq="765,verbose,memory=1215"
    local default_time="00:10:00"

    local saw_cpus_per_task=0
    local saw_mem=0
    local saw_gres=0
    local saw_gpu_freq=0
    local saw_time=0

    local -a srun_args=()
    local -a command_args=()

    while (($#)); do
        case "$1" in
            --)
                shift
                command_args=("$@")
                break
                ;;
            --cpus-per-task)
                saw_cpus_per_task=1
                srun_args+=("$1")
                shift
                if (($#)); then
                    srun_args+=("$1")
                    shift
                fi
                ;;
            --cpus-per-task=*)
                saw_cpus_per_task=1
                srun_args+=("$1")
                shift
                ;;
            --mem)
                saw_mem=1
                srun_args+=("$1")
                shift
                if (($#)); then
                    srun_args+=("$1")
                    shift
                fi
                ;;
            --mem=*)
                saw_mem=1
                srun_args+=("$1")
                shift
                ;;
            --gres)
                saw_gres=1
                srun_args+=("$1")
                shift
                if (($#)); then
                    srun_args+=("$1")
                    shift
                fi
                ;;
            --gres=*)
                saw_gres=1
                srun_args+=("$1")
                shift
                ;;
            --gpu-freq)
                saw_gpu_freq=1
                srun_args+=("$1")
                shift
                if (($#)); then
                    srun_args+=("$1")
                    shift
                fi
                ;;
            --gpu-freq=*)
                saw_gpu_freq=1
                srun_args+=("$1")
                shift
                ;;
            --time|-t)
                saw_time=1
                srun_args+=("$1")
                shift
                if (($#)); then
                    srun_args+=("$1")
                    shift
                fi
                ;;
            --time=*|-t*)
                saw_time=1
                srun_args+=("$1")
                shift
                ;;
            --partition|--qos|--account|--job-name|--output|--error|--constraint|--nodelist|--exclude|--chdir)
                srun_args+=("$1")
                shift
                if (($#)); then
                    srun_args+=("$1")
                    shift
                fi
                ;;
            --partition=*|--qos=*|--account=*|--job-name=*|--output=*|--error=*|--constraint=*|--nodelist=*|--exclude=*|--chdir=*)
                srun_args+=("$1")
                shift
                ;;
            -*)
                srun_args+=("$1")
                shift
                ;;
            *)
                command_args=("$@")
                break
                ;;
        esac
    done

    if ((${#command_args[@]} == 0)); then
        echo "usage: a5-srun [srun options] <command> [args...]" >&2
        return 2
    fi

    local -a defaults=()
    ((saw_cpus_per_task)) || defaults+=(--cpus-per-task "$default_cpus_per_task")
    ((saw_mem)) || defaults+=(--mem "$default_mem")
    ((saw_gres)) || defaults+=(--gres "$default_gres")
    ((saw_gpu_freq)) || defaults+=(--gpu-freq "$default_gpu_freq")
    ((saw_time)) || defaults+=(--time "$default_time")

    srun "${defaults[@]}" "${srun_args[@]}" "${command_args[@]}"
}
